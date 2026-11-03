//! Owned snapshots cross the Python boundary. No borrowed NumPy memory survives detach.
use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyBytes;
use std::sync::Mutex;
mod planner;

#[pyfunction]
fn plan_bytes<'py>(py:Python<'py>,data:&[u8],m:usize,k:usize,q:usize,delay:f32,late:f32,horizon:f32,
        durations:Vec<f32>,delays:Vec<f32>,complete:bool,beam:usize,branches:usize,fast:f32,age:f64)
        ->PyResult<(Bound<'py,PyBytes>,Vec<f64>)> {
    if m>4096 || k>4096 || q>32 || !(1..=4).contains(&beam) || !(3..=18).contains(&branches)
        || durations.is_empty() || durations.len()>4 || delays.is_empty() || delays.len()>12
        || data.len()!=8*(88+7*m+9*k+5*q) {
        return Err(PyValueError::new_err("planner shape or budget mismatch"));
    }
    let values:Vec<f64>=data.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap())).collect();
    if values.iter().any(|v|!v.is_finite() || v.abs()>f32::MAX as f64)
        || [delay,late,horizon,fast].iter().any(|v|!v.is_finite() || *v<=0.)
        || !age.is_finite() || age<0. || horizon>10000. || late>horizon || delay>horizon
        || durations.iter().any(|v|!v.is_finite() || *v<=0. || *v>10000.)
        || delays.iter().any(|v|!v.is_finite() || *v<0. || *v>horizon)
        || delays.windows(2).any(|v|v[0]>=v[1]) || values[4]>values[6] || values[5]>values[7]
        || values[values.len()-1]<0. {
        return Err(PyValueError::new_err("invalid planner geometry or interval"));
    }
    let (result,meta)=py.detach(move ||planner::plan(&values,m,k,q,delay,late,horizon,&durations,&delays,complete,beam,branches,fast,age));
    let bytes:Vec<u8>=result.iter().flat_map(|v|v.to_le_bytes()).collect();
    Ok((PyBytes::new(py,&bytes),meta))
}

fn floats(bytes: &[u8]) -> PyResult<Vec<f32>> {
    if bytes.len() % 4 != 0 { return Err(PyValueError::new_err("unaligned float32 buffer")); }
    let v: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    if v.iter().any(|x| !x.is_finite()) { return Err(PyValueError::new_err("non-finite geometry")); }
    Ok(v)
}

// Minimum signed L-infinity distance and continuous slab entry for a moving rectangle.
fn rectangle(r: [f32;2], v: [f32;2], e: [f32;2], length: f32) -> (f32,f32) {
    let c=[r[0]-e[0],-r[0]-e[0],r[1]-e[1],-r[1]-e[1]];
    let s=[v[0],-v[0],v[1],-v[1]];
    let gap=|t:f32| (0..4).map(|i|c[i]+s[i]*t).fold(f32::NEG_INFINITY,f32::max);
    let mut minimum=gap(0.).min(gap(length));
    for i in 0..4 { for j in i+1..4 {
        let d=s[i]-s[j]; let t=if d.abs()>1e-7 {(c[j]-c[i])/d} else {0.};
        minimum=minimum.min(gap(t.clamp(0.,length)));
    }}
    let (mut entry,mut exit)=(0_f32,length);
    for i in 0..2 {
        if v[i].abs()>1e-7 {let a=(-e[i]-r[i])/v[i];let b=(e[i]-r[i])/v[i];entry=entry.max(a.min(b));exit=exit.min(a.max(b));}
        else if r[i].abs()>e[i] {return (minimum,f32::INFINITY);}
    }
    (minimum,if entry<=exit {entry} else {f32::INFINITY})
}

fn sweep(data: &[f32], n:usize,m:usize,_k:usize,player_radius:f32)->Vec<f32> {
    let starts=&data[..2*n];let velocities=&data[2*n..4*n];
    let durations=&data[4*n..5*n];let ticks=&data[5*n..6*n];
    let low=&data[6*n..6*n+2];let high=&data[6*n+2..6*n+4];
    let circles=&data[6*n+4..6*n+4+6*m];let bodies=&data[6*n+4+6*m..];
    let mut out=Vec::with_capacity(5*n);
    for row in 0..n {
        let start=[starts[2*row],starts[2*row+1]];let vel=[velocities[2*row],velocities[2*row+1]];
        let duration=durations[row];let tick=ticks[row];
        let point=|t:f32| [(start[0]+vel[0]*t).clamp(low[0],high[0]),(start[1]+vel[1]*t).clamp(low[1],high[1])];
        let end=point(duration);
        let requested=((vel[0]*duration).powi(2)+(vel[1]*duration).powi(2)).sqrt();
        let displacement=((end[0]-start[0]).powi(2)+(end[1]-start[1]).powi(2)).sqrt();
        let blocked=if requested<1e-6 {0.} else {(1.-displacement/requested).clamp(0.,1.)};
        let mut times=[0.,duration,duration,duration];
        for axis in 0..2 {if vel[axis].abs()>1e-6 {let wall=if vel[axis]>0. {high[axis]} else {low[axis]};times[axis+1]=((wall-start[axis])/vel[axis]).clamp(0.,duration);}}
        times.sort_by(f32::total_cmp);
        let (mut minimum,mut collision)=(f32::INFINITY,f32::INFINITY);
        for piece in 0..3 {
            let begin=times[piece];let length=times[piece+1]-begin;if length<=1e-6 {continue;}
            let p=point(begin);let q=point(times[piece+1]);let pv=[(q[0]-p[0])/length,(q[1]-p[1])/length];
            let absolute=tick+begin;let absolute_end=tick+times[piece+1];
            for b in circles.chunks_exact(6) {
                let r=[b[0]+b[2]*absolute-p[0],b[1]+b[3]*absolute-p[1]];let v=[b[2]-pv[0],b[3]-pv[1]];
                let radius=b[4]+2.+b[5]*absolute_end;
                let a=v[0]*v[0]+v[1]*v[1];let dot=r[0]*v[0]+r[1]*v[1];let c=r[0]*r[0]+r[1]*r[1]-radius*radius;
                let t=(-dot/a.max(1e-12)).clamp(0.,length);
                // Match NumPy's dot + sqrt. General-purpose hypotf adds
                // overflow/subnormal handling to every candidate/obstacle pair.
                let near=[r[0]+v[0]*t,r[1]+v[1]*t];
                minimum=minimum.min((near[0]*near[0]+near[1]*near[1]).sqrt()-radius);
                let d=dot*dot-a*c;let entry=(-dot-d.max(0.).sqrt())/a.max(1e-12);
                let hit=if c<=0. {0.} else if a>1e-12 && d>=0. && entry>=0. && entry<=length {entry} else {f32::INFINITY};
                collision=collision.min(absolute+hit);
            }
            for b in bodies.chunks_exact(8) {
                let r0=[b[0]+b[2]*absolute-p[0],b[1]+b[3]*absolute-p[1]];let v0=[b[2]-pv[0],b[3]-pv[1]];
                let (s,c)=b[7].sin_cos();
                let r=[r0[0]*c+r0[1]*s,-r0[0]*s+r0[1]*c];let v=[v0[0]*c+v0[1]*s,-v0[0]*s+v0[1]*c];
                let padding=player_radius+2.+b[6]*absolute_end;
                let (clear,entry)=rectangle(r,v,[b[4]+padding,b[5]+padding],length);
                minimum=minimum.min(clear);collision=collision.min(absolute+entry);
            }
        }
        out.extend_from_slice(&[end[0],end[1],minimum,collision,blocked]);
    }
    out
}

#[pyfunction]
fn sweep_bytes<'py>(py:Python<'py>,data:&[u8],n:usize,m:usize,k:usize,player_radius:f32)->PyResult<Bound<'py,PyBytes>> {
    if n>100_000 || m>4096 || k>4096 {return Err(PyValueError::new_err("geometry budget exceeded"));}
    let data=floats(data)?;
    if data.len()!=6*n+4+6*m+8*k || !player_radius.is_finite() || player_radius<0. {return Err(PyValueError::new_err("geometry shape mismatch"));}
    if data[4*n..6*n].iter().any(|v|*v<0.) || (0..2).any(|a|data[6*n+a]>data[6*n+2+a]) {return Err(PyValueError::new_err("invalid interval"));}
    let result=py.detach(move ||sweep(&data,n,m,k,player_radius));
    let bytes:Vec<u8>=result.iter().flat_map(|v|v.to_le_bytes()).collect();
    Ok(PyBytes::new(py,&bytes))
}

// Diagnostic-only entry point. The normal collision function and its default
// GIL behavior above remain unchanged. Time pure geometry separately from
// releasing/reacquiring Python's GIL, with identical owned input snapshots.
#[pyfunction]
#[pyo3(signature=(data,n,m,k,player_radius,release_gil=true))]
fn profile_sweep_bytes<'py>(py:Python<'py>,data:&[u8],n:usize,m:usize,k:usize,
        player_radius:f32,release_gil:bool)->PyResult<(Bound<'py,PyBytes>,u64,u64)> {
    if n>100_000 || m>4096 || k>4096 {return Err(PyValueError::new_err("geometry budget exceeded"));}
    let data=floats(data)?;
    if data.len()!=6*n+4+6*m+8*k || !player_radius.is_finite() || player_radius<0. {return Err(PyValueError::new_err("geometry shape mismatch"));}
    if data[4*n..6*n].iter().any(|v|*v<0.) || (0..2).any(|a|data[6*n+a]>data[6*n+2+a]) {return Err(PyValueError::new_err("invalid interval"));}
    let compute=move|| {
        let start=std::time::Instant::now();
        let result=sweep(&data,n,m,k,player_radius);
        (result,start.elapsed().as_nanos() as u64)
    };
    let before=std::time::Instant::now();
    let (result,compute_ns)=if release_gil {py.detach(compute)} else {compute()};
    let handoff_ns=(before.elapsed().as_nanos() as u64).saturating_sub(compute_ns);
    let bytes:Vec<u8>=result.iter().flat_map(|v|v.to_le_bytes()).collect();
    Ok((PyBytes::new(py,&bytes),compute_ns,handoff_ns))
}

const NAMES:usize=10; // up,down,left,right,shift,z,x,esc,r,q; preserve existing bit positions
fn changes(held:u16,wanted:u16)->Vec<(usize,bool)> {
    let mut out=Vec::new();
    for i in 0..NAMES {if held&(1<<i)!=0 && wanted&(1<<i)==0 {out.push((i,false));}}
    for i in [4,0,1,2,3,5,6,7,8,9] {if held&(1<<i)==0 && wanted&(1<<i)!=0 {out.push((i,true));}}
    out
}

#[cfg(windows)]
mod os {
    use std::ffi::c_void;
    #[repr(C)] #[derive(Clone,Copy)] struct Keyboard {vk:u16,scan:u16,flags:u32,time:u32,extra:usize}
    #[repr(C)] #[derive(Clone,Copy)] union Event {keyboard:Keyboard,align:[usize;4]}
    #[repr(C)] #[derive(Clone,Copy)] struct Input {kind:u32,event:Event}
    #[link(name="user32")] unsafe extern "system" {fn SendInput(n:u32,inputs:*const Input,size:i32)->u32;fn GetForegroundWindow()->*mut c_void;fn GetWindowThreadProcessId(window:*mut c_void,pid:*mut u32)->u32;}
    pub fn focused(pid:u32,hwnd:usize)->bool {unsafe {let h=GetForegroundWindow();let mut p=0;GetWindowThreadProcessId(h,&mut p);h as usize==hwnd && p==pid}}
    pub fn send(changes:&[(usize,bool)])->bool {
        let inputs:Vec<Input>=changes.iter().map(|&(i,down)| {
            let scan=[0x48,0x50,0x4b,0x4d,0,0,0,0,0,0][i];
            let vk=[0,0,0,0,0x10,0x5a,0x58,0x1b,0x52,0x51][i];
            let mut event=Event{align:[0;4]};
            event.keyboard=Keyboard{vk,scan,flags:(if i<4 {9} else {0})|if down {0} else {2},time:0,extra:0};
            Input{kind:1,event}
        }).collect();
        inputs.is_empty() || unsafe {SendInput(inputs.len() as u32,inputs.as_ptr(),std::mem::size_of::<Input>() as i32)==inputs.len() as u32}
    }
    #[cfg(test)] #[test] fn input_abi(){assert_eq!(std::mem::size_of::<Input>(),40);assert_eq!(std::mem::size_of::<Keyboard>(),24);}
}

#[pyclass]
struct NativeInput {held:Mutex<u16>,pid:u32,hwnd:usize,dry_run:bool}
impl NativeInput {
    fn apply(&self,wanted:u16)->Result<(u16,usize),String> {
        let mut held=self.held.lock().map_err(|_|"input lock poisoned")?;
        let ops=changes(*held,wanted);
        #[cfg(windows)] if !self.dry_run {
            if wanted!=0 && !os::focused(self.pid,self.hwnd) {os::send(&changes(*held,0));*held=0;return Err("game is not foreground".into());}
            if !os::send(&ops) {os::send(&changes(*held|wanted,0));*held=0;return Err("SendInput incomplete; keys released".into());}
        }
        #[cfg(not(windows))] if !self.dry_run {return Err("Windows required".into());}
        *held=wanted;Ok((wanted,ops.len()))
    }
}
#[pymethods]
impl NativeInput {
    #[new] #[pyo3(signature=(pid,hwnd,dry_run=false))]
    fn new(pid:u32,hwnd:usize,dry_run:bool)->PyResult<Self>{if !dry_run && (pid==0 || hwnd==0){return Err(PyValueError::new_err("game identity required"));}Ok(Self{held:Mutex::new(0),pid,hwnd,dry_run})}
    // This is a tiny, serialized input transaction with no Python callbacks.
    // Releasing/reacquiring the GIL here lets a competing Python thread add
    // milliseconds after SendInput. Only the expensive geometry detaches.
    fn set(&self,wanted:u16)->PyResult<(u16,usize)>{if wanted>=1<<NAMES{return Err(PyValueError::new_err("unknown key"));}self.apply(wanted).map_err(PyRuntimeError::new_err)}
    #[getter] fn held(&self)->PyResult<u16>{self.held.lock().map(|v|*v).map_err(|_|PyRuntimeError::new_err("input lock poisoned"))}
}

fn checked_ranges(ranges:&[(usize,usize)])->Result<(),String> {
    if ranges.is_empty() || ranges.len()>64 {return Err("invalid read count".into());}
    let mut total=0usize;
    for &(address,size) in ranges {
        if address<0x10000 || size==0 || address.checked_add(size).is_none_or(|end|end>0x7fffffffffff) {
            return Err("invalid read range".into());
        }
        total=total.checked_add(size).ok_or("read size overflow")?;
    }
    if total>16*1024*1024 {return Err("read budget exceeded".into());}
    Ok(())
}

#[cfg(windows)]
fn read_owned(handle:usize,address:usize,size:usize)->Result<Vec<u8>,String> {
    use std::ffi::c_void;
    #[link(name="kernel32")] unsafe extern "system" {
        fn ReadProcessMemory(handle:*mut c_void,address:*const c_void,buffer:*mut c_void,size:usize,read:*mut usize)->i32;
    }
    let mut out=vec![0u8;size];let mut count=0usize;
    if unsafe {ReadProcessMemory(handle as *mut c_void,address as *const c_void,out.as_mut_ptr() as *mut c_void,size,&mut count)}==0 || count!=size {
        return Err(format!("read-only capture failed: {}",std::io::Error::last_os_error()));
    }
    Ok(out)
}
#[cfg(not(windows))]
fn read_owned(_:usize,_:usize,_:usize)->Result<Vec<u8>,String>{Err("Windows required".into())}

#[pyfunction]
fn read_many<'py>(py:Python<'py>,handle:usize,ranges:Vec<(usize,usize)>)->PyResult<(Vec<Bound<'py,PyBytes>>,u64,u64)> {
    if handle==0 {return Err(PyValueError::new_err("read handle required"));}
    checked_ranges(&ranges).map_err(PyValueError::new_err)?;
    // Requests and all result buffers are owned before detaching. No Python
    // mutable buffer or NumPy view can cross this boundary.
    let before=std::time::Instant::now();
    let (data,compute_ns)=py.detach(move|| {
        let started=std::time::Instant::now();
        let result=ranges.iter().map(|&(address,size)|read_owned(handle,address,size)).collect::<Result<Vec<_>,_>>();
        (result,started.elapsed().as_nanos() as u64)
    });
    let handoff_ns=(before.elapsed().as_nanos() as u64).saturating_sub(compute_ns);
    let data=data.map_err(PyRuntimeError::new_err)?;
    Ok((data.iter().map(|block|PyBytes::new(py,block)).collect(),compute_ns,handoff_ns))
}

#[pyfunction]
fn read_indirect_bytes<'py>(py:Python<'py>,handle:usize,pointer:usize,offset:usize,size:usize)->PyResult<Option<Bound<'py,PyBytes>>> {
    if handle==0 || offset>0x20000 || size==0 || size>0x20000-offset {return Err(PyValueError::new_err("invalid indirect read"));}
    checked_ranges(&[(pointer,8)]).map_err(PyValueError::new_err)?;
    let result=py.detach(move||->Result<Option<Vec<u8>>,String>{
        let bytes=read_owned(handle,pointer,8)?;
        let address=u64::from_le_bytes(bytes.try_into().unwrap()) as usize;
        if address==0 {return Ok(None);}
        let start=address.checked_add(offset).ok_or("pointer overflow")?;
        checked_ranges(&[(start,size)])?;
        Ok(Some(read_owned(handle,start,size)?))
    }).map_err(PyRuntimeError::new_err)?;
    Ok(result.as_ref().map(|bytes|PyBytes::new(py,bytes)))
}

#[pymodule]
fn _native_backend(m:&Bound<'_,PyModule>)->PyResult<()> {
    m.add("API_VERSION",3)?;m.add("SEARCH_API_VERSION",1)?;
    m.add_function(wrap_pyfunction!(plan_bytes,m)?)?;m.add_function(wrap_pyfunction!(sweep_bytes,m)?)?;
    m.add_function(wrap_pyfunction!(profile_sweep_bytes,m)?)?;
    m.add_function(wrap_pyfunction!(read_many,m)?)?;m.add_function(wrap_pyfunction!(read_indirect_bytes,m)?)?;
    m.add_class::<NativeInput>()?;Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn continuous_rect(){let (d,t)=rectangle([5.,0.],[-2.,0.],[1.,1.],4.);assert_eq!(t,2.);assert!(d<0.);}
    #[test] fn key_order(){assert_eq!(changes(1,48),vec![(0,false),(4,true),(5,true)]);}
    #[test] fn menu_q_keeps_existing_key_bits(){assert_eq!(changes(1<<8,1<<9),vec![(8,false),(9,true)]);}
    #[test] fn bounded_read_requests(){assert!(checked_ranges(&[(0x10000,1024)]).is_ok());assert!(checked_ranges(&[(0,1)]).is_err());assert!(checked_ranges(&[(0x10000,17*1024*1024)]).is_err());}
    #[test] fn concurrent_inputs(){let c=std::sync::Arc::new(NativeInput::new(0,0,true).unwrap());let threads:Vec<_>=(0..8).map(|i|{let c=c.clone();std::thread::spawn(move||for j in 0..1000 {c.apply(1<<((i+j)%9)).unwrap();})}).collect();for t in threads{t.join().unwrap();}c.apply(0).unwrap();assert_eq!(*c.held.lock().unwrap(),0);}
}

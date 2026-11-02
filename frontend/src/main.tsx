import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './style.css'

const root = document.getElementById('root')!
const app = <StrictMode><App /></StrictMode>
if (import.meta.env.MODE === 'benchmark') {
  void import('./renderProfile').then(({ mountProfile }) => mountProfile(root, app))
} else {
  createRoot(root).render(app)
}

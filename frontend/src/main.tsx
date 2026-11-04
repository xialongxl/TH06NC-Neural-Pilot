import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { BattleViewPage } from './BattleView'
import { LanguageProvider } from './i18n'
import './style.css'

const root = document.getElementById('root')!
const battleView = /^\/battle-view\/?$/.test(window.location.pathname)
const app = <StrictMode><LanguageProvider>{battleView ? <BattleViewPage /> : <App />}</LanguageProvider></StrictMode>
if (!battleView && import.meta.env.MODE === 'benchmark') {
  void import('./renderProfile').then(({ mountProfile }) => mountProfile(root, app))
} else {
  createRoot(root).render(app)
}

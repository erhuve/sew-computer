import Studio from './pages/Studio';
import { lazy, Suspense } from 'react';

const VisualDemo = lazy(() => import('./pages/VisualDemo'));

export default function App() {
  if (window.location.pathname === '/demo') return <Suspense fallback={<p role="status">Opening the demo…</p>}><VisualDemo /></Suspense>;
  return <Studio />;
}

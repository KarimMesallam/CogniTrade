export default function NotFound() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-slate-900 text-white">
      <div className="text-center">
        <h1 className="text-6xl font-bold mb-4">404</h1>
        <p className="text-xl mb-6">Page Not Found</p>
        <a href="/" className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors">
          Return Home
        </a>
      </div>
    </div>
  );
} 
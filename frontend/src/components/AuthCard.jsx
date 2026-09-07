export default function AuthCard({ children }) {
  return (
    <div className="min-h-full flex items-center justify-center bg-slate-200 p-4">
      <div className="w-full max-w-sm rounded-lg border border-slate-300 bg-white p-8 shadow-lg">
        <div className="mb-6 text-center">
          <h1 className="text-2xl font-bold text-primary">Rubicon</h1>
          <p className="mt-1 text-sm text-slate-500">
            Autonomous Multimodal Disaster Assessment
          </p>
        </div>
        {children}
      </div>
    </div>
  );
}
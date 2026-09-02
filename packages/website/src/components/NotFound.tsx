import { Link } from '@tanstack/react-router';

export function NotFound({ children }: { children?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 p-2">
      <div className="text-gray-600">{children || <p>The page you are looking for does not exist.</p>}</div>
      <p className="flex flex-wrap items-center gap-2">
        <button
          className="rounded-sm bg-emerald-500 px-2 py-1 text-sm font-black text-white uppercase"
          onClick={() => window.history.back()}
          type="button"
        >
          Go back
        </button>
        <Link className="rounded-sm bg-cyan-600 px-2 py-1 text-sm font-black text-white uppercase" to="/">
          Start Over
        </Link>
      </p>
    </div>
  );
}

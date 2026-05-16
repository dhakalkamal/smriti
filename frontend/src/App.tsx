import { useHealthCheck } from "./features/health/api/useHealthCheck";
import { cn } from "./lib/cn";

function App() {
  const health = useHealthCheck();

  const isHealthy = health.status === "success";
  const statusLabel =
    isHealthy
      ? "Local API online"
      : health.status === "pending"
        ? "Checking local API"
        : "Local API offline";

  return (
    <main className="min-h-screen bg-background text-foreground">
      <section className="mx-auto flex min-h-screen w-full max-w-4xl flex-col justify-center px-6 py-10">
        <div className="border-l-4 border-accent pl-5">
          <p className="text-sm font-medium uppercase tracking-normal text-muted-foreground">
            Smriti
          </p>
          <h1 className="mt-3 text-4xl font-semibold tracking-normal text-foreground">
            Frontend foundation
          </h1>
          <div className="mt-6 flex items-center gap-3" aria-live="polite">
            <span
              className={cn(
                "h-2.5 w-2.5 rounded-full",
                isHealthy
                  ? "bg-accent"
                  : health.status === "pending"
                    ? "bg-muted-foreground"
                    : "bg-danger",
              )}
              aria-hidden="true"
            />
            <p className="text-sm text-muted-foreground">{statusLabel}</p>
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;

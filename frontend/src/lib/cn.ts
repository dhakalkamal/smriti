type ClassValue = string | false | null | undefined;

export function cn(...classes: ClassValue[]): string {
  return classes.filter((className): className is string => Boolean(className)).join(" ");
}

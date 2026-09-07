/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#1E293B",
          surface: "#334155",
        },
        severity: {
          none: "#22C55E",
          moderate: "#F59E0B",
          severe: "#DC2626",
        },
        verify: "#0EA5E9",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        card: "8px",
      },
    },
  },
  plugins: [],
};
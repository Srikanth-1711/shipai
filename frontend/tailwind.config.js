/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        agent: {
          discovery: '#3B82F6', // blue
          architect: '#8B5CF6', // purple
          builder: '#10B981',   // green
          explainer: '#06B6D4', // cyan
        }
      }
    },
  },
  plugins: [],
}

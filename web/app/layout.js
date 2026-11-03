import "./globals.css";
export const metadata = { description: "Tennis Scout Agent, verified reports from database queries, not model text.", title: "Tennis Scout Agent" };
export default function RootLayout({ children }) {
  return (
    <html lang="zh">
      <body>{children}</body>
    </html>
  );
}

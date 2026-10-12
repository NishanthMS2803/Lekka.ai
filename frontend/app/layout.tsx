import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/Header";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Lekka AI | Autonomous Financial Research",
  description:
    "Multi-agent autonomous financial research platform with audited SEC filings, DCF valuations, and quantitative risk checks.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-zinc-50 text-zinc-900 dark:bg-black dark:text-zinc-100 font-sans">
        <Header />
        <div className="flex-1 flex flex-col">{children}</div>
        <footer className="border-t border-zinc-200 dark:border-zinc-800 py-6 text-center text-xs text-zinc-500 dark:text-zinc-500">
          <p>© 2026 Lekka AI. Multi-Agent Financial Research System.</p>
        </footer>
      </body>
    </html>
  );
}

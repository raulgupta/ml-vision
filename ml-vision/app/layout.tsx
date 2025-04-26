import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import Navbar from "../components/Navbar";
import { SiteFooter } from "../components/SiteFooter";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "ML Vision",
  description: "Military-grade AI platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="icon" href="/android-chrome-192x192.png" type="image/png" sizes="192x192" />
        <link rel="icon" href="/android-chrome-512x512.png" type="image/png" sizes="512x512" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/site.webmanifest" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-black min-h-screen relative`}
      >
        {/* Military gradient background with mesh overlay */}
        <div className="fixed inset-0 military-gradient">
          <div className="absolute inset-0 military-mesh opacity-20" />
          <div className="absolute inset-0 military-mesh opacity-10 scale-150 rotate-45" />
          <div className="absolute inset-0 military-mesh opacity-5 scale-200 -rotate-45" />
        </div>

        {/* Content wrapper */}
        <div className="relative z-10">
          <Navbar />
          {children}
          <SiteFooter />
        </div>
      </body>
    </html>
  );
}

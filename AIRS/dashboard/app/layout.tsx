import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AIRS Dashboard',
  description: 'AI-Driven Risk Surveillance for Portfolio Protection',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  )
}

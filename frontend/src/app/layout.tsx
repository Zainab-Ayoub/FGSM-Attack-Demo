export const metadata = {
  title: 'FGSM Demo',
  description: 'Adversarial attack demo (FGSM) with FastAPI backend',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: 'system-ui, sans-serif', margin: 24 }}>{children}</body>
    </html>
  );
}
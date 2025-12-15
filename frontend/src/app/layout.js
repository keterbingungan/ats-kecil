import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata = {
  title: "Intelligent ATS - AI-Powered Applicant Tracking",
  description: "AI-powered Applicant Tracking System. Upload CVs, extract skills automatically using DistilBERT NER, and find the best candidates for your role.",
  keywords: ["ATS", "Applicant Tracking", "CV Parser", "NER", "Skill Extraction", "Machine Learning"],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${inter.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}

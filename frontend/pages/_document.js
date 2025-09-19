import { Html, Head, Main, NextScript } from 'next/document'
import Document from 'next/document'

export default class MyDocument extends Document {
  render() {
    return (
      <Html lang="en">
        <Head>
          {/* ✅ Tailwind CDN */}
          <script src="https://cdn.tailwindcss.com"></script>
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}

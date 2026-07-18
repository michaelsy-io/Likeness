import { handleUpload } from '@vercel/blob/client';

const MAX_IMAGE_BYTES = 25 * 1024 * 1024;
const uploadToken = process.env.PUBLIC_INTAKE_READ_WRITE_TOKEN || process.env.BLOB_READ_WRITE_TOKEN;

// Vercel's Web Handler signature provides a standard Request object, which
// `handleUpload` needs to generate a browser-upload token.
export default {
  async fetch(request) {
    // A deliberately non-sensitive diagnostic used by the browser before it
    // attempts an upload. It confirms deployment configuration without ever
    // exposing a token or Blob store identifier.
    if (request.method === 'GET') {
      return Response.json({
        configured: Boolean(uploadToken),
        service: 'vercel-blob-client-upload',
      }, {
        status: uploadToken ? 200 : 503,
        headers: { 'Cache-Control': 'no-store' },
      });
    }

    if (request.method !== 'POST') {
      return Response.json({ error: 'Method not allowed.' }, { status: 405 });
    }

    try {
      if (!uploadToken) {
        throw new Error('The Public Vercel Blob read-write token is not configured.');
      }

      const body = await request.json();
      const result = await handleUpload({
        body,
        request,
        token: uploadToken,
        onBeforeGenerateToken: async (pathname) => {
          if (!pathname.startsWith('likeness-intake/')) {
            throw new Error('Invalid upload destination.');
          }
          return {
            allowedContentTypes: ['image/jpeg', 'image/png', 'image/webp'],
            maximumSizeInBytes: MAX_IMAGE_BYTES,
            addRandomSuffix: true,
            validUntil: Date.now() + 5 * 60 * 1000,
          };
        },
        onUploadCompleted: async () => {
          // The analysis request verifies the public Blob URL before use.
        },
      });
      return Response.json(result);
    } catch (error) {
      console.error('Blob client-upload authorization failed:', error);
      return Response.json(
        { error: error instanceof Error ? error.message : 'Unable to authorize image upload.' },
        { status: 400 },
      );
    }
  },
};

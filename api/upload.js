import { handleUpload } from '@vercel/blob/client';

const MAX_IMAGE_BYTES = 25 * 1024 * 1024;
const uploadToken = process.env.PUBLIC_INTAKE_READ_WRITE_TOKEN || process.env.BLOB_READ_WRITE_TOKEN;

export default async function handler(request) {
  if (request.method !== 'POST') {
    return Response.json({ error: 'Method not allowed.' }, { status: 405 });
  }

  try {
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
    return Response.json(
      { error: error instanceof Error ? error.message : 'Unable to authorize image upload.' },
      { status: 400 },
    );
  }
}

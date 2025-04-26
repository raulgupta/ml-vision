import { NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(request: Request) {
  const { prompt, negativePrompt } = await request.json();
  
  // Add hidden CV-focused instructions but keep them separate from the user's prompt
  const enhancedPrompt = `${prompt}. Create this as a clean, high-contrast image suitable for computer vision applications with good edge definition and feature points. The image should have even lighting, clear subject separation, and minimal noise.`;
  
  try {
    const response = await openai.images.generate({
      model: "dall-e-3",
      prompt: enhancedPrompt + (negativePrompt ? ` (Avoid: ${negativePrompt})` : ''),
      n: 1,
      size: "1024x1024",
    });
    
    // Extract just the user-facing part of the revised prompt
    // This removes our CV-specific instructions from what the user sees
    const originalRevisedPrompt = response.data[0].revised_prompt || '';
    
    // Get the image URL from OpenAI
    const openaiImageUrl = response.data[0].url || '';
    
    // Create a proxied URL to avoid CORS issues
    const origin = request.headers.get('origin') || process.env.VERCEL_URL || 'http://localhost:4000';
    const proxiedImageUrl = `${origin}/api/proxy-image?url=${encodeURIComponent(openaiImageUrl)}`;
    
    return NextResponse.json({ 
      imageUrl: proxiedImageUrl,
      revisedPrompt: prompt // Return the original prompt instead of the revised one
    });
  } catch (error: any) {
    console.error('Error generating image:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to generate image' },
      { status: 500 }
    );
  }
}

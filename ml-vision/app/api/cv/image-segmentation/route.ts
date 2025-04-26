import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    console.log('Image segmentation request received');
    
    // Forward the request to our Python service
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/segment-image`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...body,
        confidence_threshold: body.confidence_threshold || 0.2,
        use_yolo: body.use_yolo !== undefined ? body.use_yolo : true, // Default to using YOLO
        yolo_confidence: body.yolo_confidence || 0.2, // Default YOLO confidence threshold
      }),
    });
    
    console.log(`Backend response status: ${response.status}`);
    
    if (!response.ok) {
      throw new Error(`Python service returned ${response.status}`);
    }
    
    // Log response headers to check content type and size
    const headers: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      headers[key] = value;
    });
    console.log('Response headers:', headers);
    
    // Get the raw text first to see if it's valid JSON
    const responseText = await response.text();
    console.log(`Response size: ${responseText.length} bytes`);
    
    let data;
    try {
      // Try to parse the JSON
      data = JSON.parse(responseText);
      console.log('Response successfully parsed as JSON');
    } catch (parseError) {
      console.error('JSON parsing error:', parseError);
      console.error('First 200 chars of response:', responseText.substring(0, 200));
      throw new Error('Failed to parse response from Python service');
    }
    
    // Check if there was an error from the Python service
    if (data.error) {
      console.error('Python service returned error:', data.error, data.message);
      return NextResponse.json(
        { error: data.error, message: data.message },
        { status: 500 }
      );
    }
    
    // Validate expected data structure
    if (!data.processed_image) {
      console.error('Missing processed_image in response');
      return NextResponse.json(
        { error: 'Invalid response format from Python service' },
        { status: 500 }
      );
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Enhanced image segmentation error:', error);
    return NextResponse.json(
      { error: 'Failed to process image', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}

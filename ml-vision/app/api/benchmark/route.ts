import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  console.log('Benchmark API route called');
  try {
    // Forward the request to our Python service
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    console.log(`Sending request to backend at ${backendUrl}/benchmark`);
    const response = await fetch(`${backendUrl}/benchmark`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      console.error(`Backend returned error: ${response.status} ${response.statusText}`);
      throw new Error(`Python service returned ${response.status}`);
    }
    
    console.log('Received response from backend');
    const data = await response.json();
    console.log('Parsed response data:', data);
    return NextResponse.json(data);
  } catch (error) {
    console.error('Benchmark error:', error);
    return NextResponse.json(
      { error: 'Failed to run benchmark' },
      { status: 500 }
    );
  }
}

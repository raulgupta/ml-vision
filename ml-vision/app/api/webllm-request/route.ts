import { NextRequest } from 'next/server';

// Add CORS headers to response
function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

// Queue for requests and responses
const requestQueue = new Map<string, { message: string; timestamp: number }>();
const responseQueue = new Map<string, { response: string; timestamp: number }>();

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { message } = body;

    if (!message) {
      return Response.json(
        { error: 'Message is required' },
        { status: 400, headers: corsHeaders() }
      );
    }

    // Generate unique request ID
    const requestId = Math.random().toString(36).substring(2);
    
    // Add request to queue
    requestQueue.set(requestId, {
      message,
      timestamp: Date.now()
    });

    // Wait for response with timeout
    let attempts = 0;
    const maxAttempts = 60; // 30 seconds with 500ms intervals
    
    while (attempts < maxAttempts) {
      const response = responseQueue.get(requestId);
      if (response) {
        // Clean up
        responseQueue.delete(requestId);
        requestQueue.delete(requestId);
        
        return Response.json(
          { response: response.response },
          { headers: corsHeaders() }
        );
      }
      
      await new Promise(resolve => setTimeout(resolve, 500));
      attempts++;
    }

    // If we timeout, clean up and return error
    requestQueue.delete(requestId);
    return Response.json(
      { error: 'Request timed out waiting for WebLLM response' },
      { status: 504, headers: corsHeaders() }
    );

  } catch (error) {
    console.error('Error in WebLLM request:', error);
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to process request' },
      { status: 500, headers: corsHeaders() }
    );
  }
}

// Endpoint to get pending requests
export async function GET(req: NextRequest) {
  try {
    // Get the oldest pending request
    const entries = Array.from(requestQueue.entries());
    const oldestEntry = entries.reduce<[string, any] | null>((oldest, current) => {
      if (!oldest || current[1].timestamp < oldest[1].timestamp) {
        return current;
      }
      return oldest;
    }, null);

    if (!oldestEntry) {
      return Response.json(
        { status: 'no_requests' },
        { headers: corsHeaders() }
      );
    }

    const [id, request] = oldestEntry;
    
    return Response.json({
      id,
      message: request.message
    }, { headers: corsHeaders() });

  } catch (error) {
    console.error('Error getting pending requests:', error);
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to get pending requests' },
      { status: 500, headers: corsHeaders() }
    );
  }
}

// Endpoint to submit responses
export async function PUT(req: NextRequest) {
  try {
    const body = await req.json();
    const { id, response } = body;

    if (!id || !response) {
      return Response.json(
        { error: 'ID and response are required' },
        { status: 400, headers: corsHeaders() }
      );
    }

    // Add response to queue
    responseQueue.set(id, {
      response,
      timestamp: Date.now()
    });

    return Response.json(
      { status: 'success' },
      { headers: corsHeaders() }
    );

  } catch (error) {
    console.error('Error submitting response:', error);
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to submit response' },
      { status: 500, headers: corsHeaders() }
    );
  }
}

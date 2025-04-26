import { NextRequest } from 'next/server';
import { webLLMService } from '../../../components/WebLLMService';

// Add CORS headers to response
function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

// Handle OPTIONS request for CORS preflight
export async function OPTIONS(request: NextRequest) {
  return new Response(null, {
    status: 204,
    headers: corsHeaders(),
  });
}

interface Message {
  role: string;
  content: string;
}

interface WebLLMRequest {
  messages: Message[];
  stream?: boolean;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { messages, stream = false } = body as WebLLMRequest;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return Response.json(
        { error: 'Messages array is required and must not be empty' },
        { 
          status: 400,
          headers: corsHeaders()
        }
      );
    }

    // Get the last user message
    const userMessage = messages.filter(m => m.role === 'user').pop();
    if (!userMessage) {
      return Response.json(
        { error: 'No user message found in messages array' },
        { 
          status: 400,
          headers: corsHeaders()
        }
      );
    }

    console.log('Processing WebLLM request:', { messages });

    // Handle streaming response if requested
    if (stream) {
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async start(controller) {
          try {
            await webLLMService.generateResponse(
              userMessage.content,
              (chunk: string) => {
                // Send each chunk as a JSON object
                const data = JSON.stringify({ chunk }) + '\n';
                controller.enqueue(encoder.encode(data));
              }
            );
            controller.close();
          } catch (error) {
            const errorMsg = JSON.stringify({ 
              error: error instanceof Error ? error.message : 'Unknown error'
            }) + '\n';
            controller.enqueue(encoder.encode(errorMsg));
            controller.close();
          }
        }
      });

      return new Response(stream, {
        headers: {
          ...corsHeaders(),
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive'
        }
      });
    }

    // Handle non-streaming response (backward compatibility)
    try {
      const response = await webLLMService.generateResponse(userMessage.content);
      console.log('WebLLM response generated:', response);
      
      return Response.json(
        { response },
        { headers: corsHeaders() }
      );
    } catch (error) {
      console.error('WebLLM generation error:', error);
      return Response.json(
        { 
          error: error instanceof Error ? error.message : 'Failed to generate response',
          details: error instanceof Error ? error.stack : undefined
        },
        { 
          status: 500,
          headers: corsHeaders()
        }
      );
    }

  } catch (error) {
    console.error('WebLLM API error:', error);
    return Response.json(
      { 
        error: error instanceof Error ? error.message : 'Failed to process request',
        details: error instanceof Error ? error.stack : undefined
      },
      { 
        status: 500,
        headers: corsHeaders()
      }
    );
  }
}

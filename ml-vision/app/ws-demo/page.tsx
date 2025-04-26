'use client';

import { useState, useEffect, useRef } from 'react';

export default function WebSocketDemo() {
    const [messages, setMessages] = useState<string[]>([]);
    const [inputMessage, setInputMessage] = useState('');
    const [connected, setConnected] = useState(false);
    const ws = useRef<WebSocket | null>(null);

    useEffect(() => {
        ws.current = new WebSocket('ws://localhost:9000/ws');

        ws.current.onopen = () => {
            console.log('Connected to WebSocket');
            setConnected(true);
        };

        ws.current.onmessage = (event) => {
            setMessages(prev => [...prev, event.data]);
        };

        ws.current.onclose = () => {
            console.log('Disconnected from WebSocket');
            setConnected(false);
        };

        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, []);

    const sendMessage = () => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN && inputMessage) {
            ws.current.send(inputMessage);
            setInputMessage('');
        }
    };

    return (
        <main className="min-h-screen w-full relative overflow-hidden">
            {/* Military Background Pattern */}
            <div className="absolute inset-0 military-gradient">
                <div className="absolute inset-0 military-mesh opacity-20" />
                <div className="absolute inset-0 military-mesh opacity-10 scale-150 rotate-45" />
                <div className="absolute inset-0 military-mesh opacity-5 scale-200 -rotate-45" />
            </div>

            <div className="container mx-auto px-4 py-8">
                <div className="mt-40 md:mt-48">
                    <div className="max-w-3xl mx-auto">
                        <div className="flex justify-between items-center mb-12">
                            <h2 className="text-2xl font-mono text-white/40">WEBSOCKET DEMO</h2>
                            <div className="px-4 py-1.5 md:px-6 md:py-2 bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg">
                                <div className="flex items-center gap-2">
                                    <div className={`w-2 h-2 rounded-full ${connected ? 'bg-white/90' : 'bg-white/20'}`}></div>
                                    <span className={`font-venus text-base md:text-lg ${connected ? 'text-white/90' : 'text-white/40'}`}>
                                        {connected ? 'CONNECTED' : 'DISCONNECTED'}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Main Card */}
                        <div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 shadow-[0_0_15px_rgba(255,255,255,0.02)] space-y-6">
                            {/* Message Input */}
                            <div className="space-y-2">
                                <h3 className="text-sm font-mono text-white/40">SEND MESSAGE</h3>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={inputMessage}
                                        onChange={(e) => setInputMessage(e.target.value)}
                                        onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                                        placeholder="Type a message..."
                                        className="flex-1 bg-white/[0.02] border border-white/[0.03] rounded-lg px-4 py-2 font-mono text-white/90 placeholder:text-white/20 focus:outline-none focus:border-white/10"
                                    />
                                    <button
                                        onClick={sendMessage}
                                        className="px-6 py-2 bg-white/[0.02] border border-white/[0.03] rounded-lg font-mono text-white/90 hover:bg-white/[0.05] transition-all"
                                    >
                                        SEND
                                    </button>
                                </div>
                            </div>

                            {/* Messages Display */}
                            <div className="space-y-2">
                                <h3 className="text-sm font-mono text-white/40">MESSAGES</h3>
                                <div className="h-96 overflow-y-auto space-y-2 pr-2">
                                    {messages.map((msg, index) => (
                                        <div 
                                            key={index} 
                                            className="bg-white/[0.02] border border-white/[0.03] rounded-lg p-4"
                                        >
                                            <p className="font-mono text-white/90">{msg}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}

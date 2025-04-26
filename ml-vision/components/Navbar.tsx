'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';

const navItems = [
  { id: '01', name: 'Home', href: '/' },
  { id: '02', name: 'Computer Vision', href: '/cv' },
  { id: '03', name: 'NeRF Showcase', href: '/showcase' },
  { id: '04', name: 'Benchmark', href: '/benchmark' },
  { id: '05', name: 'Resume', href: '/resume' },
];

const MenuIcon = ({ isOpen }: { isOpen: boolean }) => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path 
      d={isOpen ? "M18 6L6 18M6 6L18 18" : "M4 6H20M4 12H20M4 18H20"} 
      stroke="currentColor" 
      strokeWidth="1.5" 
      strokeLinecap="round" 
      strokeLinejoin="round"
    />
  </svg>
);

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isVisible, setIsVisible] = useState(true);
  const [prevScrollPos, setPrevScrollPos] = useState(0);

  const handleEscKey = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') setIsMenuOpen(false);
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollPos = window.scrollY;
      setScrolled(currentScrollPos > 0);
      
      // Show navbar at the top of the page
      if (currentScrollPos <= 0) {
        setIsVisible(true);
        setPrevScrollPos(currentScrollPos);
        return;
      }

      // Determine scroll direction and update visibility
      const isScrollingUp = prevScrollPos > currentScrollPos;
      setIsVisible(isScrollingUp);
      setPrevScrollPos(currentScrollPos);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [prevScrollPos]);

  useEffect(() => {
    document.body.style.overflow = isMenuOpen ? 'hidden' : 'unset';
    if (isMenuOpen) window.addEventListener('keydown', handleEscKey);
    else window.removeEventListener('keydown', handleEscKey);
    return () => window.removeEventListener('keydown', handleEscKey);
  }, [isMenuOpen, handleEscKey]);

  const commonButtonStyles = "flex items-center justify-center w-11 h-11 rounded-full bg-white/[0.04] backdrop-blur-[8px] hover:bg-white/[0.08] transition-all duration-200 active:scale-95";
  const commonLinkStyles = "text-white/80 hover:text-white transition-all duration-200";

  return (
    <>
      <div className={`fixed top-0 left-0 right-0 z-40 transition-transform duration-300 transform ${
        isVisible ? 'translate-y-0' : '-translate-y-[150%]'
      }`}>
        <header className={`w-full pt-4 transition-all duration-300 ${
          scrolled ? 'bg-white/5 backdrop-blur-md' : ''
        }`}>
          <nav className="relative flex items-center justify-center w-full">
            {/* Logo */}
            <Link 
              href="/"
              className={`hidden md:flex ${commonButtonStyles} mr-5`}
            >
              <span className="text-2xl text-white/70" style={{ fontFamily: 'serif' }}>✧</span>
            </Link>

            {/* Desktop Navigation Items Container */}
            <div className="hidden md:flex items-center px-3 py-2 bg-white/[0.04] backdrop-blur-[8px] rounded-2xl group/nav min-w-[580px]">
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`${commonLinkStyles} px-7 py-1.5 text-[16px] group-hover/nav:opacity-50 hover:!opacity-100 mr-2 last:mr-0`}
                >
                  {item.name}
                </Link>
              ))}
            </div>

            {/* Mobile Navigation */}
            <div className="flex md:hidden w-full px-8">
              <div className="flex items-center justify-between w-full">
                {/* Mobile Logo */}
                <Link 
                  href="/"
                  className={commonButtonStyles}
                >
                  <span className="text-2xl text-white/70" style={{ fontFamily: 'serif' }}>✧</span>
                </Link>

                {/* Mobile Menu Toggle Button */}
                <button
                  onClick={() => setIsMenuOpen(!isMenuOpen)}
                  className={commonButtonStyles}
                  aria-label="Toggle menu"
                >
                  <MenuIcon isOpen={isMenuOpen} />
                </button>
              </div>
            </div>
          </nav>
        </header>

        {/* Updates notification */}
        <div className="w-full max-w-[450px] mx-auto px-4 md:px-0 mt-6">
          <div className="group flex items-center justify-between px-6 py-2.5 bg-white/[0.02] hover:bg-white/[0.03] backdrop-blur-[8px] rounded-2xl text-[14px] text-white/40 transition-all duration-300 cursor-pointer">
            <span>Discover more about our latest updates.</span>
            <div className="ml-4 px-3 py-1 rounded-full bg-[#0066ff]/5 transition-all duration-300 group-hover:translate-x-1">
              <svg 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                xmlns="http://www.w3.org/2000/svg" 
                className="text-[#0066ff] transition-all duration-300"
              >
                <path 
                  d="M4 12H20M20 12L13 5M20 12L13 19" 
                  stroke="currentColor" 
                  strokeWidth="1.5" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Menu Overlay - Positioned outside the transform container */}
      {isMenuOpen && (
        <div 
          className="fixed inset-0 bg-black/95 backdrop-blur-md z-50"
          onClick={(e) => e.target === e.currentTarget && setIsMenuOpen(false)}
        >
          <button
            onClick={() => setIsMenuOpen(false)}
            className={`${commonButtonStyles} fixed top-4 right-8 z-[60] ring-1 ring-white/10 hover:ring-white/20`}
            aria-label="Close menu"
          >
            <MenuIcon isOpen={true} />
          </button>

          <div className="fixed inset-0 flex flex-col items-start justify-center p-8">
            <nav className="w-full space-y-6">
              {navItems.map((item, index) => (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={() => setIsMenuOpen(false)}
                  className={`${commonLinkStyles} flex items-center space-x-4 text-3xl font-light`}
                  style={{
                    animationDelay: `${index * 100}ms`,
                    animation: 'menuItemEnter 0.5s ease forwards',
                  }}
                >
                  <span className="text-sm text-white/40">{item.id}</span>
                  <span>{item.name}</span>
                </Link>
              ))}
            </nav>
          </div>
        </div>
      )}
    </>
  );
}

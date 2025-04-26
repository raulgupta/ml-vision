# Gun AI Design System & Core Styles

## Color System

### Core Colors
```css
--background: #0a0a0a     /* Main background */
--foreground: #ededed     /* Primary text color */
--military-dark: #1a1f1a  /* Military theme dark */
--military-mid: #2a2f2a   /* Military theme mid */
--military-light: #4a524a /* Military theme light */
--accent: #3f4f3f         /* Accent color */
```

## Typography

### Font Families
- `font-venus`: Primary display font for metrics and numbers
- `font-mono`: Used for labels and technical text

### Text Opacity Variants
- `text-white/90`: Primary text
- `text-white/40`: Secondary text, labels
- `text-white/[0.03]`: Subtle borders

## Core Components

### Military Background Pattern
The signature background pattern consists of three layers:
```jsx
<div className="absolute inset-0 military-gradient">
  <div className="absolute inset-0 military-mesh opacity-20" />
  <div className="absolute inset-0 military-mesh opacity-10 scale-150 rotate-45" />
  <div className="absolute inset-0 military-mesh opacity-5 scale-200 -rotate-45" />
</div>
```

### Card Components
Common card styling pattern:
```jsx
<div className="bg-white/[0.02] backdrop-blur-sm border border-white/[0.03] rounded-lg p-6 shadow-[0_0_15px_rgba(255,255,255,0.02)]">
  {/* Content */}
</div>
```

## Utility Classes

### Spacing
- `space-y-12`: Large vertical spacing between sections
- `space-y-2`: Small vertical spacing between elements
- `gap-2`, `gap-4`: Flex and grid gaps

### Layout
- `container mx-auto`: Page container
- `px-4 py-8`: Standard padding
- `grid grid-cols-2`: Two-column grid layout
- `min-h-screen`: Full viewport height
- `relative z-10`: Content layering

### Backdrop Effects
```css
.backdrop-blur-sm {
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
```

### Custom Gradients
```css
.military-gradient {
  background: linear-gradient(
    to bottom right,
    var(--military-dark),
    var(--military-mid),
    var(--military-dark)
  );
}

.military-mesh {
  background-image: radial-gradient(
    circle at center,
    var(--military-light) 0%,
    var(--accent) 25%,
    transparent 70%
  );
}
```

## Common Text Patterns

### Headings
```jsx
<h2 className="text-sm font-mono text-white/40">SECTION TITLE</h2>
```

### Metrics Display
```jsx
<div className="flex items-baseline gap-2">
  <span className="text-4xl font-venus text-white/90">{value}</span>
  <span className="text-sm text-white/40">unit label</span>
</div>
```

## Animations & Transitions

### Transitions
```css
.transition-all {
  transition-property: all;
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
  transition-duration: 300ms;
}
```

### Chart Animations
```javascript
animation: {
  duration: 1000,
  easing: 'easeInOutQuart'
}
```

## Best Practices

1. Use opacity variants with white for text hierarchy:
   - 90% for primary content
   - 40% for secondary content
   - 3% for subtle borders

2. Layer backdrop blur effects with semi-transparent backgrounds for depth:
   ```jsx
   className="bg-white/[0.02] backdrop-blur-sm"
   ```

3. Maintain consistent spacing using the space-y utility:
   - `space-y-12` for major sections
   - `space-y-2` for related elements

4. Use the military gradient and mesh pattern for background effects

5. Implement the Venus font for numerical displays and metrics

6. Use monospace font for labels and technical information

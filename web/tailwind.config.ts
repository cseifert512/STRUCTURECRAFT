import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Organic/Natural palette
        cream: {
          50: '#FDFCFA',
          100: '#FAF9F6',
          200: '#F5F0E8',
          300: '#EDE5D8',
        },
        sage: {
          50: '#F4F7F4',
          100: '#E8EFE8',
          200: '#C5D5C5',
          300: '#A3BBA3',
          400: '#87A878',
          500: '#6B8F5B',
          600: '#4A6741',
          700: '#3D5536',
          800: '#2F422A',
        },
        clay: {
          50: '#FBF8F6',
          100: '#F7F0EC',
          200: '#EDD9CE',
          300: '#E3C2B0',
          400: '#D4A574',
          500: '#C4846C',
          600: '#A66B52',
          700: '#8A5742',
        },
        forest: {
          600: '#4A6741',
          700: '#3D5536',
          800: '#2C3E2D',
          900: '#1F2B1F',
        },
        stone: {
          400: '#9CA89C',
          500: '#6B7B6C',
          600: '#5A695B',
        },
      },
      fontFamily: {
        serif: ['var(--font-dm-serif)', 'Georgia', 'serif'],
        sans: ['var(--font-jakarta)', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 2px 15px -3px rgba(44, 62, 45, 0.07), 0 10px 20px -2px rgba(44, 62, 45, 0.04)',
        'soft-lg': '0 10px 40px -10px rgba(44, 62, 45, 0.12), 0 20px 25px -5px rgba(44, 62, 45, 0.05)',
        'glow': '0 0 30px -5px rgba(135, 168, 120, 0.3)',
      },
      borderRadius: {
        '4xl': '2rem',
      },
      backgroundImage: {
        'grain': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E\")",
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
    },
  },
  plugins: [],
}

export default config


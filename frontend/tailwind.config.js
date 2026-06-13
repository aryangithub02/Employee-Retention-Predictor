/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'signal-orange': '#ff682c',
        'sienna-bronze': '#816729',
        carbon: '#202020',
        graphite: '#4d4d4d',
        slate: '#828282',
        fog: '#f5f5f5',
        mist: '#efefef',
        chalk: '#e8e8e8',
        paper: '#ffffff',
      },
      fontFamily: {
        polysans: ['"Space Grotesk"', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        inter: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
      },
      fontSize: {
        caption: ['12px', { lineHeight: '1.5' }],
        body: ['16px', { lineHeight: '1.38' }],
        'body-lg': ['18px', { lineHeight: '1.33' }],
        subheading: ['32px', { lineHeight: '1.19', letterSpacing: '-0.64px' }],
        heading: ['40px', { lineHeight: '1.13', letterSpacing: '-0.8px' }],
        display: ['66px', { lineHeight: '0.91', letterSpacing: '-1.32px' }],
      },
      spacing: {
        '8': '8px',
        '12': '12px',
        '16': '16px',
        '20': '20px',
        '36': '36px',
        '40': '40px',
        '60': '60px',
        '140': '140px',
      },
      borderRadius: {
        'cards': '8px',
        'inputs': '8px',
        'buttons': '20px',
        'tags': '20px',
        'navpill': '200px',
        'full': '200px',
      },
      maxWidth: {
        page: '1200px',
      },
      boxShadow: {
        'card': '0 1px 3px rgba(32, 32, 32, 0.04), 0 4px 12px rgba(32, 32, 32, 0.03)',
        'card-hover': '0 2px 6px rgba(32, 32, 32, 0.06), 0 8px 24px rgba(32, 32, 32, 0.05)',
        'navpill': '0 1px 3px rgba(32, 32, 32, 0.04), 0 4px 12px rgba(32, 32, 32, 0.03)',
      },
      animation: {
        'fadeIn': 'fadeIn 0.4s ease-out',
        'slideIn': 'slideIn 0.4s ease-out',
        'scaleIn': 'scaleIn 0.35s ease-out',
        'slideUp': 'slideUp 0.4s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideIn: {
          '0%': { opacity: '0', transform: 'translateX(-20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
};

export default function PostureLogo({ className = 'h-20 w-20 text-white' }) {
  return (
    <svg
      className={className}
      viewBox="0 0 160 160"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="posture-logo-glow" x1="20" y1="20" x2="140" y2="140" gradientUnits="userSpaceOnUse">
          <stop stopColor="currentColor" stopOpacity="0.75" />
          <stop offset="1" stopColor="currentColor" stopOpacity="0.2" />
        </linearGradient>
        <radialGradient id="posture-logo-head" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse">
          <stop stopColor="currentColor" stopOpacity="0.9" />
          <stop offset="1" stopColor="currentColor" stopOpacity="0.35" />
        </radialGradient>
      </defs>

      <g stroke="url(#posture-logo-glow)" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round">
        <path
          d="M52 120c24-10 38-10 58-4 12 4 18 2 22-4"
          opacity="0.35"
          strokeWidth="10"
          strokeLinecap="round"
        />
        <path
          d="M80 66c-8 10-12 26-22 46"
          strokeWidth="9"
          opacity="0.6"
          strokeLinecap="round"
        />
        <path
          d="M90 110c3-18 4-36 16-48 6-6 10-20-4-24-10-3-18 4-26 18"
          strokeWidth="9"
        />
      </g>

      <circle
        cx="86"
        cy="44"
        r="18"
        fill="url(#posture-logo-head)"
        stroke="currentColor"
        strokeWidth="6"
        opacity="0.9"
      />

      <path
        d="M54 136c32 14 58 10 78-8"
        stroke="currentColor"
        strokeOpacity="0.25"
        strokeWidth="6"
        strokeLinecap="round"
      />
    </svg>
  )
}


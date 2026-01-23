# StructureCraft Web Interface

Beautiful, real-time 3D canopy designer built with Next.js, React Three Fiber, and Tailwind CSS.

## Features

- **Real-time 3D visualization** using React Three Fiber
- **Live parameter updates** with debounced API calls
- **Organic/natural design aesthetic** with custom Tailwind theme
- **Force visualization** with tension/compression coloring
- **Export functionality** for CSV cut lists and JSON models

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+ (for the backend API)

### Setup

1. **Install frontend dependencies:**
   ```bash
   cd web
   npm install
   ```

2. **Start the backend API** (in a separate terminal):
   ```bash
   cd api
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

3. **Start the frontend:**
   ```bash
   cd web
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Tech Stack

- **Framework:** Next.js 14 with App Router
- **3D Graphics:** React Three Fiber + Drei
- **Styling:** Tailwind CSS with custom organic theme
- **State:** Zustand
- **Animations:** Framer Motion
- **UI Components:** Radix UI primitives

## Design System

### Color Palette

- **Cream** - Warm off-white backgrounds
- **Sage** - Primary green tones
- **Clay/Terracotta** - Warm accent colors
- **Forest** - Deep green text colors

### Typography

- **Headings:** DM Serif Display
- **Body:** Plus Jakarta Sans

## Project Structure

```
web/
├── src/
│   ├── app/
│   │   ├── layout.tsx      # Root layout with fonts
│   │   ├── page.tsx        # Main designer page
│   │   └── globals.css     # Global styles
│   ├── components/
│   │   ├── Canvas3D.tsx    # 3D viewer
│   │   ├── ControlPanel.tsx
│   │   ├── MetricsCard.tsx
│   │   ├── ExportMenu.tsx
│   │   └── ui/             # Reusable UI primitives
│   ├── lib/
│   │   ├── api.ts          # API client
│   │   ├── types.ts        # TypeScript types
│   │   └── utils.ts        # Utilities
│   └── store/
│       └── designStore.ts  # Zustand state
├── tailwind.config.ts      # Custom theme
└── package.json
```


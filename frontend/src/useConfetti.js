import { useCallback, useEffect, useRef } from "react";

const COLORS = ["#7c5cff", "#22d3ee", "#ffd93d", "#b8f2e6", "#c77dff"];
const PIECES = 100;
const DURATION_FRAMES = 160;

/**
 * Canvas confetti burst, torn down when it finishes or the component unmounts.
 *
 * The teardown matters: an unmount mid-burst has to cancel the animation frame
 * and drop the resize listener, or both outlive the component.
 */
export function useConfetti() {
  const cleanupRef = useRef(null);

  useEffect(() => () => cleanupRef.current?.(), []);

  return useCallback(() => {
    if (cleanupRef.current) return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;

    const canvas = document.createElement("canvas");
    canvas.className = "confetti";
    canvas.setAttribute("aria-hidden", "true");
    document.body.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const pieces = Array.from({ length: PIECES }, (_, i) => ({
      x: Math.random() * canvas.width,
      y: -20 - Math.random() * canvas.height * 0.4,
      size: 4 + Math.random() * 6,
      vx: -2 + Math.random() * 4,
      vy: 2 + Math.random() * 3,
      rotation: Math.random() * Math.PI,
      spin: -0.1 + Math.random() * 0.2,
      color: COLORS[i % COLORS.length],
    }));

    let frame = 0;
    let raf = 0;

    const stop = () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      canvas.remove();
      cleanupRef.current = null;
    };
    cleanupRef.current = stop;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const p of pieces) {
        p.x += p.vx;
        p.y += p.vy;
        p.rotation += p.spin;
        if (p.y > canvas.height + 20) {
          p.y = -20;
          p.x = Math.random() * canvas.width;
        }
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rotation);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
        ctx.restore();
      }
      if (++frame < DURATION_FRAMES) raf = requestAnimationFrame(draw);
      else stop();
    };

    raf = requestAnimationFrame(draw);
  }, []);
}

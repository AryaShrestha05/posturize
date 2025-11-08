import { useEffect, useRef } from 'react'

function SmokeyCursor({
  simulationResolution = 128,
  dyeResolution = 1440,
  captureResolution = 512,
  densityDissipation = 3.5,
  velocityDissipation = 2,
  pressure = 0.1,
  pressureIterations = 20,
  curl = 3,
  splatRadius = 0.2,
  splatForce = 6000,
  enableShading = true,
  colorUpdateSpeed = 10,
  backgroundColor = { r: 0, g: 0.06, b: 0.14 },
  transparent = true,
}) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const pointers = [createPointer()]
    const config = {
      SIM_RESOLUTION: simulationResolution,
      DYE_RESOLUTION: dyeResolution,
      CAPTURE_RESOLUTION: captureResolution,
      DENSITY_DISSIPATION: densityDissipation,
      VELOCITY_DISSIPATION: velocityDissipation,
      PRESSURE: pressure,
      PRESSURE_ITERATIONS: pressureIterations,
      CURL: curl,
      SPLAT_RADIUS: splatRadius,
      SPLAT_FORCE: splatForce,
      SHADING: enableShading,
      COLOR_UPDATE_SPEED: colorUpdateSpeed,
      PAUSED: false,
      BACK_COLOR: backgroundColor,
      TRANSPARENT: transparent,
    }

    const glContext = getWebGLContext(canvas)
    if (!glContext) return
    const { gl, ext } = glContext

    if (!ext.supportLinearFiltering) {
      config.DYE_RESOLUTION = 256
      config.SHADING = false
    }

    const baseVertexShader = compileShader(
      gl,
      gl.VERTEX_SHADER,
      `
        precision highp float;
        attribute vec2 aPosition;
        varying vec2 vUv;
        varying vec2 vL;
        varying vec2 vR;
        varying vec2 vT;
        varying vec2 vB;
        uniform vec2 texelSize;
        void main () {
            vUv = aPosition * 0.5 + 0.5;
            vL = vUv - vec2(texelSize.x, 0.0);
            vR = vUv + vec2(texelSize.x, 0.0);
            vT = vUv + vec2(0.0, texelSize.y);
            vB = vUv - vec2(0.0, texelSize.y);
            gl_Position = vec4(aPosition, 0.0, 1.0);
        }
      `
    )

    const copyShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision mediump float;
        precision mediump sampler2D;
        varying highp vec2 vUv;
        uniform sampler2D uTexture;
        void main () {
            gl_FragColor = texture2D(uTexture, vUv);
        }
      `
    )

    const clearShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision mediump float;
        precision mediump sampler2D;
        varying highp vec2 vUv;
        uniform sampler2D uTexture;
        uniform float value;
        void main () {
            gl_FragColor = value * texture2D(uTexture, vUv);
        }
      `
    )

    const displayShaderSource = `
      precision highp float;
      precision highp sampler2D;
      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform sampler2D uTexture;
      uniform vec2 texelSize;
      void main () {
          vec3 c = texture2D(uTexture, vUv).rgb;
          #ifdef SHADING
              vec3 lc = texture2D(uTexture, vL).rgb;
              vec3 rc = texture2D(uTexture, vR).rgb;
              vec3 tc = texture2D(uTexture, vT).rgb;
              vec3 bc = texture2D(uTexture, vB).rgb;
              float dx = length(rc) - length(lc);
              float dy = length(tc) - length(bc);
              vec3 n = normalize(vec3(dx, dy, length(texelSize)));
              vec3 l = vec3(0.0, 0.0, 1.0);
              float diffuse = clamp(dot(n, l) + 0.7, 0.7, 1.0);
              c *= diffuse;
          #endif
          float a = max(c.r, max(c.g, c.b));
          gl_FragColor = vec4(c, a);
      }
    `

    const splatShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision highp float;
        precision highp sampler2D;
        varying vec2 vUv;
        uniform sampler2D uTarget;
        uniform float aspectRatio;
        uniform vec3 color;
        uniform vec2 point;
        uniform float radius;
        void main () {
            vec2 p = vUv - point.xy;
            p.x *= aspectRatio;
            vec3 splat = exp(-dot(p, p) / radius) * color;
            vec3 base = texture2D(uTarget, vUv).xyz;
            gl_FragColor = vec4(base + splat, 1.0);
        }
      `
    )

    const advectionShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision highp float;
        precision highp sampler2D;
        varying vec2 vUv;
        uniform sampler2D uVelocity;
        uniform sampler2D uSource;
        uniform vec2 texelSize;
        uniform vec2 dyeTexelSize;
        uniform float dt;
        uniform float dissipation;
        vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
            vec2 st = uv / tsize - 0.5;
            vec2 iuv = floor(st);
            vec2 fuv = fract(st);
            vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
            vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
            vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
            vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
            return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
        }
        void main () {
            #ifdef MANUAL_FILTERING
                vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
                vec4 result = bilerp(uSource, coord, dyeTexelSize);
            #else
                vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
                vec4 result = texture2D(uSource, coord);
            #endif
            float decay = 1.0 + dissipation * dt;
            gl_FragColor = result / decay;
        }
      `,
      ext.supportLinearFiltering ? null : ['MANUAL_FILTERING']
    )

    const divergenceShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision mediump float;
        precision mediump sampler2D;
        varying highp vec2 vUv;
        varying highp vec2 vL;
        varying highp vec2 vR;
        varying highp vec2 vT;
        varying highp vec2 vB;
        uniform sampler2D uVelocity;
        void main () {
            float L = texture2D(uVelocity, vL).x;
            float R = texture2D(uVelocity, vR).x;
            float T = texture2D(uVelocity, vT).y;
            float B = texture2D(uVelocity, vB).y;
            vec2 C = texture2D(uVelocity, vUv).xy;
            if (vL.x < 0.0) { L = -C.x; }
            if (vR.x > 1.0) { R = -C.x; }
            if (vT.y > 1.0) { T = -C.y; }
            if (vB.y < 0.0) { B = -C.y; }
            float div = 0.5 * (R - L + T - B);
            gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
        }
      `
    )

    const curlShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision mediump float;
        precision mediump sampler2D;
        varying highp vec2 vUv;
        varying highp vec2 vL;
        varying highp vec2 vR;
        varying highp vec2 vT;
        varying highp vec2 vB;
        uniform sampler2D uVelocity;
        void main () {
            float L = texture2D(uVelocity, vL).y;
            float R = texture2D(uVelocity, vR).y;
            float T = texture2D(uVelocity, vT).x;
            float B = texture2D(uVelocity, vB).x;
            float vorticity = R - L - T + B;
            gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
        }
      `
    )

    const vorticityShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision highp float;
        precision highp sampler2D;
        varying vec2 vUv;
        varying vec2 vL;
        varying vec2 vR;
        varying vec2 vT;
        varying vec2 vB;
        uniform sampler2D uVelocity;
        uniform sampler2D uCurl;
        uniform float curl;
        uniform float dt;
        void main () {
            float L = texture2D(uCurl, vL).x;
            float R = texture2D(uCurl, vR).x;
            float T = texture2D(uCurl, vT).x;
            float B = texture2D(uCurl, vB).x;
            float C = texture2D(uCurl, vUv).x;
            vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
            force /= length(force) + 0.0001;
            force *= curl * C;
            force.y *= -1.0;
            vec2 velocity = texture2D(uVelocity, vUv).xy;
            velocity += force * dt;
            velocity = min(max(velocity, -1000.0), 1000.0);
            gl_FragColor = vec4(velocity, 0.0, 1.0);
        }
      `
    )

    const pressureShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision mediump float;
        precision mediump sampler2D;
        varying highp vec2 vUv;
        varying highp vec2 vL;
        varying highp vec2 vR;
        varying highp vec2 vT;
        varying highp vec2 vB;
        uniform sampler2D uPressure;
        uniform sampler2D uDivergence;
        void main () {
            float L = texture2D(uPressure, vL).x;
            float R = texture2D(uPressure, vR).x;
            float T = texture2D(uPressure, vT).x;
            float B = texture2D(uPressure, vB).x;
            float divergence = texture2D(uDivergence, vUv).x;
            float pressure = (L + R + B + T - divergence) * 0.25;
            gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
        }
      `
    )

    const gradientSubtractShader = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision mediump float;
        precision mediump sampler2D;
        varying highp vec2 vUv;
        varying highp vec2 vL;
        varying highp vec2 vR;
        varying highp vec2 vT;
        varying highp vec2 vB;
        uniform sampler2D uPressure;
        uniform sampler2D uVelocity;
        void main () {
            float L = texture2D(uPressure, vL).x;
            float R = texture2D(uPressure, vR).x;
            float T = texture2D(uPressure, vT).x;
            float B = texture2D(uPressure, vB).x;
            vec2 velocity = texture2D(uVelocity, vUv).xy;
            velocity.xy -= vec2(R - L, T - B);
            gl_FragColor = vec4(velocity, 0.0, 1.0);
        }
      `
    )

    const blit = createBlitter(gl)

    let dye
    let velocity
    let divergence
    let curlFBO
    let pressureFBO

    const copyProgram = new Program(gl, baseVertexShader, copyShader)
    const clearProgram = new Program(gl, baseVertexShader, clearShader)
    const splatProgram = new Program(gl, baseVertexShader, splatShader)
    const advectionProgram = new Program(gl, baseVertexShader, advectionShader)
    const divergenceProgram = new Program(gl, baseVertexShader, divergenceShader)
    const curlProgram = new Program(gl, baseVertexShader, curlShader)
    const vorticityProgram = new Program(gl, baseVertexShader, vorticityShader)
    const pressureProgram = new Program(gl, baseVertexShader, pressureShader)
    const gradientSubtractProgram = new Program(gl, baseVertexShader, gradientSubtractShader)
    const displayMaterial = new Material(gl, baseVertexShader, displayShaderSource)

    initFramebuffers()
    updateKeywords()

    let lastUpdateTime = Date.now()
    let colorUpdateTimer = 0
    let animationId = requestAnimationFrame(updateFrame)

    function updateFrame() {
      const dt = calcDeltaTime()
      if (resizeCanvas()) initFramebuffers()
      updateColors(dt)
      applyInputs()
      step(dt)
      render(null)
      animationId = requestAnimationFrame(updateFrame)
    }

    function calcDeltaTime() {
      const now = Date.now()
      let dt = (now - lastUpdateTime) / 1000
      dt = Math.min(dt, 0.016666)
      lastUpdateTime = now
      return dt
    }

    function resizeCanvas() {
      const width = scaleByPixelRatio(canvas.clientWidth)
      const height = scaleByPixelRatio(canvas.clientHeight)
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
        return true
      }
      return false
    }

    function updateColors(dt) {
      colorUpdateTimer += dt * config.COLOR_UPDATE_SPEED
      if (colorUpdateTimer >= 1) {
        colorUpdateTimer = wrap(colorUpdateTimer, 0, 1)
        pointers.forEach((pointer) => {
          pointer.color = generateColor()
        })
      }
    }

    function applyInputs() {
      pointers.forEach((pointer) => {
        if (!pointer.moved) return
        pointer.moved = false
        splatPointer(pointer)
      })
    }

    function step(dt) {
      gl.disable(gl.BLEND)

      curlProgram.bind()
      gl.uniform2f(curlProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY)
      gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.read.attach(0))
      blit(curlFBO)

      vorticityProgram.bind()
      gl.uniform2f(vorticityProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY)
      gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.read.attach(0))
      gl.uniform1i(vorticityProgram.uniforms.uCurl, curlFBO.attach(1))
      gl.uniform1f(vorticityProgram.uniforms.curl, config.CURL)
      gl.uniform1f(vorticityProgram.uniforms.dt, dt)
      blit(velocity.write)
      velocity.swap()

      divergenceProgram.bind()
      gl.uniform2f(divergenceProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY)
      gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.read.attach(0))
      blit(divergence)

      clearProgram.bind()
      gl.uniform1i(clearProgram.uniforms.uTexture, pressureFBO.read.attach(0))
      gl.uniform1f(clearProgram.uniforms.value, config.PRESSURE)
      blit(pressureFBO.write)
      pressureFBO.swap()

      pressureProgram.bind()
      gl.uniform2f(pressureProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY)
      gl.uniform1i(pressureProgram.uniforms.uDivergence, divergence.attach(0))
      for (let i = 0; i < config.PRESSURE_ITERATIONS; i += 1) {
        gl.uniform1i(pressureProgram.uniforms.uPressure, pressureFBO.read.attach(1))
        blit(pressureFBO.write)
        pressureFBO.swap()
      }

      gradientSubtractProgram.bind()
      gl.uniform2f(gradientSubtractProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY)
      gl.uniform1i(gradientSubtractProgram.uniforms.uPressure, pressureFBO.read.attach(0))
      gl.uniform1i(gradientSubtractProgram.uniforms.uVelocity, velocity.read.attach(1))
      blit(velocity.write)
      velocity.swap()

      advectionProgram.bind()
      gl.uniform2f(advectionProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY)
      if (!ext.supportLinearFiltering)
        gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, velocity.texelSizeX, velocity.texelSizeY)

      const velocityId = velocity.read.attach(0)
      gl.uniform1i(advectionProgram.uniforms.uVelocity, velocityId)
      gl.uniform1i(advectionProgram.uniforms.uSource, velocityId)
      gl.uniform1f(advectionProgram.uniforms.dt, dt)
      gl.uniform1f(advectionProgram.uniforms.dissipation, config.VELOCITY_DISSIPATION)
      blit(velocity.write)
      velocity.swap()

      if (!ext.supportLinearFiltering)
        gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, dye.texelSizeX, dye.texelSizeY)

      gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.attach(0))
      gl.uniform1i(advectionProgram.uniforms.uSource, dye.read.attach(1))
      gl.uniform1f(advectionProgram.uniforms.dissipation, config.DENSITY_DISSIPATION)
      blit(dye.write)
      dye.swap()
    }

    function render(target) {
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
      gl.enable(gl.BLEND)
      drawDisplay(target)
    }

    function drawDisplay(target) {
      const width = target == null ? gl.drawingBufferWidth : target.width
      const height = target == null ? gl.drawingBufferHeight : target.height
      displayMaterial.bind()
      if (config.SHADING) gl.uniform2f(displayMaterial.uniforms.texelSize, 1.0 / width, 1.0 / height)
      gl.uniform1i(displayMaterial.uniforms.uTexture, dye.read.attach(0))
      blit(target)
    }

    function splatPointer(pointer) {
      const dx = pointer.deltaX * config.SPLAT_FORCE
      const dy = pointer.deltaY * config.SPLAT_FORCE
      splat(pointer.texcoordX, pointer.texcoordY, dx, dy, pointer.color)
    }

    function splat(x, y, dx, dy, color) {
      splatProgram.bind()
      gl.uniform1i(splatProgram.uniforms.uTarget, velocity.read.attach(0))
      gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height)
      gl.uniform2f(splatProgram.uniforms.point, x, y)
      gl.uniform3f(splatProgram.uniforms.color, dx, dy, 0.0)
      gl.uniform1f(splatProgram.uniforms.radius, correctRadius(config.SPLAT_RADIUS / 100.0))
      blit(velocity.write)
      velocity.swap()

      gl.uniform1i(splatProgram.uniforms.uTarget, dye.read.attach(0))
      gl.uniform3f(splatProgram.uniforms.color, color.r, color.g, color.b)
      blit(dye.write)
      dye.swap()
    }

    function correctRadius(radius) {
      let aspectRatio = canvas.width / canvas.height
      if (aspectRatio > 1) radius *= aspectRatio
      return radius
    }

    function initFramebuffers() {
      const simRes = getResolution(gl, config.SIM_RESOLUTION)
      const dyeRes = getResolution(gl, config.DYE_RESOLUTION)
      const texType = ext.halfFloatTexType
      const rgba = ext.formatRGBA
      const rg = ext.formatRG
      const r = ext.formatR
      const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST

      gl.disable(gl.BLEND)

      dye = createDoubleFBO(gl, dyeRes.width, dyeRes.height, rgba.internalFormat, rgba.format, texType, filtering)
      velocity = createDoubleFBO(gl, simRes.width, simRes.height, rg.internalFormat, rg.format, texType, filtering)
      divergence = createFBO(gl, simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST)
      curlFBO = createFBO(gl, simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST)
      pressureFBO = createDoubleFBO(gl, simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST)
    }

    function updateKeywords() {
      const keywords = []
      if (config.SHADING) keywords.push('SHADING')
      displayMaterial.setKeywords(keywords)
    }

    const pointerDown = (event) => {
      const pointer = pointers[0]
      const posX = scaleByPixelRatio(event.clientX)
      const posY = scaleByPixelRatio(event.clientY)
      updatePointerDown(pointer, -1, posX, posY, canvas)
    }

    const pointerMove = (event) => {
      const pointer = pointers[0]
      const posX = scaleByPixelRatio(event.clientX)
      const posY = scaleByPixelRatio(event.clientY)
      updatePointerMove(pointer, posX, posY, canvas)
    }

    const pointerUp = () => {
      updatePointerUp(pointers[0])
    }

    const touchStart = (event) => {
      Array.from(event.changedTouches).forEach((touch) => {
        const pointer = pointers[0]
        const posX = scaleByPixelRatio(touch.clientX)
        const posY = scaleByPixelRatio(touch.clientY)
        updatePointerDown(pointer, touch.identifier, posX, posY, canvas)
      })
    }

    const touchMove = (event) => {
      Array.from(event.changedTouches).forEach((touch) => {
        const pointer = pointers[0]
        const posX = scaleByPixelRatio(touch.clientX)
        const posY = scaleByPixelRatio(touch.clientY)
        updatePointerMove(pointer, posX, posY, canvas)
      })
    }

    const touchEnd = () => {
      updatePointerUp(pointers[0])
    }

    window.addEventListener('pointerdown', pointerDown)
    window.addEventListener('pointermove', pointerMove)
    window.addEventListener('pointerup', pointerUp)
    window.addEventListener('pointercancel', pointerUp)

    window.addEventListener('touchstart', touchStart, { passive: true })
    window.addEventListener('touchmove', touchMove, { passive: true })
    window.addEventListener('touchend', touchEnd)
    window.addEventListener('touchcancel', touchEnd)

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener('pointerdown', pointerDown)
      window.removeEventListener('pointermove', pointerMove)
      window.removeEventListener('pointerup', pointerUp)
      window.removeEventListener('pointercancel', pointerUp)
      window.removeEventListener('touchstart', touchStart)
      window.removeEventListener('touchmove', touchMove)
      window.removeEventListener('touchend', touchEnd)
      window.removeEventListener('touchcancel', touchEnd)
    }
  }, [
    simulationResolution,
    dyeResolution,
    captureResolution,
    densityDissipation,
    velocityDissipation,
    pressure,
    pressureIterations,
    curl,
    splatRadius,
    splatForce,
    enableShading,
    colorUpdateSpeed,
    backgroundColor,
    transparent,
  ])

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 50,
        pointerEvents: 'none',
        background: transparent
          ? 'none'
          : 'radial-gradient(circle at center, rgba(32, 90, 175, 0.18) 0%, rgba(4, 12, 28, 0.9) 65%, rgba(0, 4, 10, 1) 100%)',
        mixBlendMode: 'screen',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: '100vw',
          height: '100vh',
          display: 'block',
        }}
      />
    </div>
  )
}

function createPointer() {
  return {
    id: -1,
    texcoordX: 0,
    texcoordY: 0,
    prevTexcoordX: 0,
    prevTexcoordY: 0,
    deltaX: 0,
    deltaY: 0,
    down: false,
    moved: false,
    color: generateColor(),
  }
}

function generateColor() {
  const brightness = 0.12
  return { r: brightness, g: brightness, b: brightness }
}

function hslToRgb(h, s, l) {
  if (s === 0) {
    return { r: l, g: l, b: l }
  }
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1
    if (t > 1) t -= 1
    if (t < 1 / 6) return p + (q - p) * 6 * t
    if (t < 1 / 2) return q
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
    return p
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  const r = hue2rgb(p, q, h + 1 / 3)
  const g = hue2rgb(p, q, h)
  const b = hue2rgb(p, q, h - 1 / 3)
  return { r, g, b }
}

function getWebGLContext(canvas) {
  const params = {
    alpha: true,
    depth: false,
    stencil: false,
    antialias: false,
    preserveDrawingBuffer: false,
  }
  let gl = canvas.getContext('webgl2', params)
  const isWebGL2 = !!gl
  if (!isWebGL2) {
    gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params)
  }
  if (!gl) return null

  let halfFloat
  let supportLinearFiltering
  if (isWebGL2) {
    gl.getExtension('EXT_color_buffer_float')
    supportLinearFiltering = gl.getExtension('OES_texture_float_linear')
  } else {
    halfFloat = gl.getExtension('OES_texture_half_float')
    supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear')
  }

  const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat && halfFloat.HALF_FLOAT_OES
  const formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType)
  const formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType)
  const formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType)

  return {
    gl,
    ext: {
      formatRGBA,
      formatRG,
      formatR,
      halfFloatTexType,
      supportLinearFiltering,
    },
  }
}

function getSupportedFormat(gl, internalFormat, format, type) {
  if (!supportRenderTextureFormat(gl, internalFormat, format, type)) {
    switch (internalFormat) {
      case gl.R16F:
        return getSupportedFormat(gl, gl.RG16F, gl.RG, type)
      case gl.RG16F:
        return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type)
      default:
        return null
    }
  }
  return { internalFormat, format }
}

function supportRenderTextureFormat(gl, internalFormat, format, type) {
  const texture = gl.createTexture()
  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null)
  const fbo = gl.createFramebuffer()
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
  return status === gl.FRAMEBUFFER_COMPLETE
}

function compileShader(gl, type, source, keywords) {
  const shader = gl.createShader(type)
  gl.shaderSource(shader, addKeywords(source, keywords))
  gl.compileShader(shader)
  return shader
}

function addKeywords(source, keywords) {
  if (!keywords || keywords.length === 0) return source
  let directives = ''
  keywords.forEach((keyword) => {
    directives += `#define ${keyword}\n`
  })
  return directives + source
}

class Program {
  constructor(gl, vertexShader, fragmentShader) {
    this.gl = gl
    this.program = createProgram(gl, vertexShader, fragmentShader)
    this.uniforms = getUniforms(gl, this.program)
  }

  bind() {
    this.gl.useProgram(this.program)
  }
}

class Material {
  constructor(gl, vertexShader, fragmentShaderSource) {
    this.gl = gl
    this.vertexShader = vertexShader
    this.fragmentShaderSource = fragmentShaderSource
    this.programs = new Map()
    this.uniforms = {}
    this.activeProgram = null
  }

  setKeywords(keywords) {
    const hash = keywords.join(',')
    if (!this.programs.has(hash)) {
      const fragmentShader = compileShader(this.gl, this.gl.FRAGMENT_SHADER, this.fragmentShaderSource, keywords)
      const program = createProgram(this.gl, this.vertexShader, fragmentShader)
      this.programs.set(hash, program)
    }
    const nextProgram = this.programs.get(hash)
    if (nextProgram === this.activeProgram) return
    this.uniforms = getUniforms(this.gl, nextProgram)
    this.activeProgram = nextProgram
  }

  bind() {
    this.gl.useProgram(this.activeProgram)
  }
}

function createProgram(gl, vertexShader, fragmentShader) {
  const program = gl.createProgram()
  gl.attachShader(program, vertexShader)
  gl.attachShader(program, fragmentShader)
  gl.linkProgram(program)
  return program
}

function getUniforms(gl, program) {
  const uniforms = {}
  const uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS)
  for (let i = 0; i < uniformCount; i += 1) {
    const name = gl.getActiveUniform(program, i).name
    uniforms[name] = gl.getUniformLocation(program, name)
  }
  return uniforms
}

function createBlitter(gl) {
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer())
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW)
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer())
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW)
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
  gl.enableVertexAttribArray(0)
  return (target, clear) => {
    if (target == null) {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    } else {
      gl.viewport(0, 0, target.width, target.height)
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo)
    }
    if (clear) {
      gl.clearColor(0.0, 0.0, 0.0, 1.0)
      gl.clear(gl.COLOR_BUFFER_BIT)
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)
  }
}

function createFBO(gl, w, h, internalFormat, format, type, param) {
  gl.activeTexture(gl.TEXTURE0)
  const texture = gl.createTexture()
  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null)
  const fbo = gl.createFramebuffer()
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
  gl.viewport(0, 0, w, h)
  gl.clear(gl.COLOR_BUFFER_BIT)
  const texelSizeX = 1.0 / w
  const texelSizeY = 1.0 / h
  return {
    texture,
    fbo,
    width: w,
    height: h,
    texelSizeX,
    texelSizeY,
    attach(id) {
      gl.activeTexture(gl.TEXTURE0 + id)
      gl.bindTexture(gl.TEXTURE_2D, texture)
      return id
    },
  }
}

function createDoubleFBO(gl, w, h, internalFormat, format, type, param) {
  let read = createFBO(gl, w, h, internalFormat, format, type, param)
  let write = createFBO(gl, w, h, internalFormat, format, type, param)
  return {
    get read() {
      return read
    },
    set read(value) {
      read = value
    },
    get write() {
      return write
    },
    set write(value) {
      write = value
    },
    width: w,
    height: h,
    texelSizeX: 1.0 / w,
    texelSizeY: 1.0 / h,
    swap() {
      const temp = read
      read = write
      write = temp
    },
  }
}

function updatePointerDown(pointer, id, posX, posY, canvas) {
  pointer.id = id
  pointer.down = true
  pointer.moved = false
  const width = canvas.width || window.innerWidth
  const height = canvas.height || window.innerHeight
  pointer.texcoordX = posX / width
  pointer.texcoordY = 1.0 - posY / height
  pointer.prevTexcoordX = pointer.texcoordX
  pointer.prevTexcoordY = pointer.texcoordY
  pointer.deltaX = 0
  pointer.deltaY = 0
  pointer.color = generateColor()
}

function updatePointerMove(pointer, posX, posY, canvas) {
  pointer.prevTexcoordX = pointer.texcoordX
  pointer.prevTexcoordY = pointer.texcoordY
  const width = canvas.width || window.innerWidth
  const height = canvas.height || window.innerHeight
  pointer.texcoordX = posX / width
  pointer.texcoordY = 1.0 - posY / height
  pointer.deltaX = (pointer.texcoordX - pointer.prevTexcoordX) * 0.6
  pointer.deltaY = (pointer.texcoordY - pointer.prevTexcoordY) * 0.6
  pointer.moved = Math.abs(pointer.deltaX) > 0 || Math.abs(pointer.deltaY) > 0
}

function updatePointerUp(pointer) {
  pointer.down = false
}

function getResolution(gl, resolution) {
  let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight
  if (aspectRatio < 1) aspectRatio = 1.0 / aspectRatio
  const min = Math.round(resolution)
  const max = Math.round(resolution * aspectRatio)
  if (gl.drawingBufferWidth > gl.drawingBufferHeight) {
    return { width: max, height: min }
  }
  return { width: min, height: max }
}

function scaleByPixelRatio(input) {
  const pixelRatio = window.devicePixelRatio || 1
  return Math.floor(input * pixelRatio)
}

function wrap(value, min, max) {
  const range = max - min
  if (range === 0) return min
  return ((value - min) % range) + min
}

export default SmokeyCursor


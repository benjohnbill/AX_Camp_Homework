import streamlit.components.v1 as components
import json
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _json_default(value: Any):
    """Convert non-JSON-native values to safe serializable forms."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _project_log(row: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    content = str(row.get("content") or "").strip()
    if not content:
        return None
    created_at = row.get("created_at")
    if created_at is None:
        created_at = row.get("timestamp")

    return {
        "id": str(row.get("id") or ""),
        "content": content,
        "meta_type": str(row.get("meta_type") or ""),
        "created_at": _json_default(created_at) if created_at is not None else "",
    }


def _project_core(row: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    content = str(row.get("content") or "").strip()
    if not content:
        return None
    created_at = row.get("created_at")
    if created_at is None:
        created_at = row.get("timestamp")

    return {
        "id": str(row.get("id") or ""),
        "content": content,
        "meta_type": str(row.get("meta_type") or ""),
        "created_at": _json_default(created_at) if created_at is not None else "",
    }


def _prepare_3d_payload(logs: Iterable[Any], cores: Iterable[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    safe_logs: List[Dict[str, Any]] = []
    for row in logs or []:
        projected = _project_log(row)
        if projected:
            safe_logs.append(projected)

    safe_cores: List[Dict[str, Any]] = []
    for row in cores or []:
        projected = _project_core(row)
        if projected:
            safe_cores.append(projected)

    return safe_logs, safe_cores

def render_3d_universe(logs, cores):
    safe_logs, safe_cores = _prepare_3d_payload(logs, cores)
    logs_json = json.dumps(safe_logs, ensure_ascii=False, default=_json_default)
    cores_json = json.dumps(safe_cores, ensure_ascii=False, default=_json_default)

    html_string = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background-color: #000; font-family: 'Courier New', monospace; user-select: none; }}
            #info-panel {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                right: 20px;
                background: rgba(0, 30, 60, 0.85);
                border: 1px solid #0ff;
                border-left: 4px solid #0ff;
                color: #0ff;
                padding: 20px;
                border-radius: 4px;
                pointer-events: none;
                display: none;
                box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
                transition: opacity 0.3s ease;
                backdrop-filter: blur(5px);
            }}
            #crosshair {{
                position: absolute;
                top: 50%;
                left: 50%;
                width: 30px;
                height: 30px;
                transform: translate(-50%, -50%);
                pointer-events: none;
                border: 1px solid rgba(0, 255, 255, 0.3);
                border-radius: 50%;
                transition: transform 0.2s;
            }}
            #crosshair::after {{
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 4px;
                height: 4px;
                background: #0ff;
                transform: translate(-50%, -50%);
                border-radius: 50%;
                box-shadow: 0 0 8px #0ff;
            }}
            .hud-text {{
                position: absolute;
                top: 20px;
                left: 20px;
                color: rgba(0, 255, 255, 0.8);
                font-size: 13px;
                line-height: 1.5;
                text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
                pointer-events: none;
                letter-spacing: 2px;
            }}
            .controls-hint {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                color: rgba(255, 255, 255, 0.5);
                font-size: 11px;
                text-align: right;
                pointer-events: none;
                line-height: 1.6;
                letter-spacing: 1px;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div class="hud-text">
            S Y S T E M : A N T I G R A V I T Y<br>
            A N O M A L Y   S C A N N E R   O N L I N E<br>
            <span style="color:#e94560;">ENTROPY: NOMINAL</span>
        </div>
        <div class="controls-hint">
            [W,A,S,D] NAVIGATE<br>
            [MOUSE DRAG] ROTATE<br>
            [SHIFT + W] BOOST DIVE<br>
            (APPROACH NODE TO READ)
        </div>
        <div id="crosshair"></div>
        <div id="info-panel">
            <h3 id="info-title" style="margin-top:0; font-size:14px; text-transform: uppercase; letter-spacing: 2px; color: #fff;">Node</h3>
            <p id="info-desc" style="font-size:16px; margin-bottom:0; line-height:1.6; font-style: italic;"></p>
        </div>
        
        <script>
            const logsData = {logs_json};
            const coresData = {cores_json};
            
            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x0a0a1a, 0.003); // Deep Space Fog
            
            const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.1, 2000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);
            
            // Lighting
            const ambient = new THREE.AmbientLight(0x222233);
            scene.add(ambient);
            const pointLight = new THREE.PointLight(0xffffff, 1, 1000);
            scene.add(pointLight);

            const interactables = [];
            
            // Master Core
            if (coresData && coresData.length > 0) {{
                const coreGeo = new THREE.SphereGeometry(25, 32, 32);
                const coreMat = new THREE.MeshBasicMaterial({{ color: 0xe94560, wireframe: true, transparent: true, opacity: 0.8 }});
                const coreMesh = new THREE.Mesh(coreGeo, coreMat);
                coreMesh.userData = {{ isCore: true, content: coresData[0].content, title: "CORE CONSTITUTION" }};
                scene.add(coreMesh);
                interactables.push(coreMesh);
                
                const innerGeo = new THREE.SphereGeometry(23, 32, 32);
                const innerMat = new THREE.MeshBasicMaterial({{ color: 0xffaaaa, transparent: true, opacity: 0.3 }});
                const innerMesh = new THREE.Mesh(innerGeo, innerMat);
                scene.add(innerMesh);
            }}

            const nodeGeo = new THREE.SphereGeometry(1.5, 16, 16);
            
            // Randomly scatter logs in an orbit around the Core
            logsData.forEach((log) => {{
                if (!log.content || log.content.length < 5) return;

                const radius = 50 + Math.random() * 450;
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos((Math.random() * 2) - 1);
                
                const x = radius * Math.sin(phi) * Math.cos(theta);
                const y = radius * Math.sin(phi) * Math.sin(theta);
                const z = radius * Math.cos(phi);
                
                const isViolation = log.meta_type && log.meta_type.toLowerCase().includes('violation');
                const color = isViolation ? 0xff3333 : (Math.random() > 0.5 ? 0x00ffff : 0x0088ff);
                
                const mat = new THREE.MeshBasicMaterial({{ color: color, transparent: true, opacity: 0.7 }});
                const mesh = new THREE.Mesh(nodeGeo, mat);
                mesh.position.set(x, y, z);
                
                mesh.userData = {{ 
                    content: log.content, 
                    title: isViolation ? `[SABOTEUR LOG] - ${{log.created_at ? log.created_at.substring(0,10) : ''}}` : `[OBSERVATION] - ${{log.created_at ? log.created_at.substring(0,10) : ''}}`
                }};
                scene.add(mesh);
                interactables.push(mesh);
            }});

            const pGeo = new THREE.BufferGeometry();
            const pCount = 3000;
            const posArray = new Float32Array(pCount * 3);
            for(let i=0; i<pCount*3; i++) {{
                posArray[i] = (Math.random() - 0.5) * 1200;
            }}
            pGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
            const pMat = new THREE.PointsMaterial({{ size: 0.4, color: 0xffffff, transparent: true, opacity: 0.4 }});
            const particles = new THREE.Points(pGeo, pMat);
            scene.add(particles);

            camera.position.z = 250;
            camera.position.y = 80;
            camera.lookAt(0,0,0);

            const keyState = {{ w:false, a:false, s:false, d:false, shift:false }};
            window.addEventListener('keydown', (e) => {{
                if(keyState[e.key.toLowerCase()] !== undefined) keyState[e.key.toLowerCase()] = true;
            }});
            window.addEventListener('keyup', (e) => {{
                if(keyState[e.key.toLowerCase()] !== undefined) keyState[e.key.toLowerCase()] = false;
            }});

            let isDragging = false;
            let prevX = 0, prevY = 0;
            const euler = new THREE.Euler(0, 0, 0, 'YXZ');
            
            renderer.domElement.addEventListener('mousedown', (e) => {{ isDragging = true; prevX = e.clientX; prevY = e.clientY; }});
            window.addEventListener('mouseup', () => isDragging = false);
            window.addEventListener('mousemove', (e) => {{
                if(isDragging) {{
                    const dx = e.clientX - prevX;
                    const dy = e.clientY - prevY;
                    prevX = e.clientX; prevY = e.clientY;
                    
                    euler.setFromQuaternion(camera.quaternion);
                    euler.y -= dx * 0.003; euler.x -= dy * 0.003;
                    euler.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, euler.x));
                    camera.quaternion.setFromEuler(euler);
                }}
            }});
            
            // Touch control for mobile panning
            renderer.domElement.addEventListener('touchstart', (e) => {{
                if(e.touches.length > 0) {{
                    isDragging = true;
                    prevX = e.touches[0].clientX;
                    prevY = e.touches[0].clientY;
                }}
            }}, {{passive: false}});
            renderer.domElement.addEventListener('touchend', () => isDragging = false);
            renderer.domElement.addEventListener('touchmove', (e) => {{
                if(isDragging && e.touches.length > 0) {{
                    e.preventDefault();
                    const dx = e.touches[0].clientX - prevX;
                    const dy = e.touches[0].clientY - prevY;
                    prevX = e.touches[0].clientX;
                    prevY = e.touches[0].clientY;
                    
                    euler.setFromQuaternion(camera.quaternion);
                    euler.y -= dx * 0.005; euler.x -= dy * 0.005;
                    euler.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, euler.x));
                    camera.quaternion.setFromEuler(euler);
                    
                    // Auto forward on double touch or just slow forward on mobile dragging
                    keyState.w = true; // Auto move forward while dragging screen
                    setTimeout(() => {{ keyState.w = false; }}, 500);
                }}
            }}, {{passive: false}});

            const infoPanel = document.getElementById('info-panel');
            const infoTitle = document.getElementById('info-title');
            const infoDesc = document.getElementById('info-desc');
            const crosshair = document.getElementById('crosshair');

            const velocity = new THREE.Vector3();
            const direction = new THREE.Vector3();
            const clock = new THREE.Clock();

            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});

            let lastHovered = null;

            function animate() {{
                requestAnimationFrame(animate);
                const delta = clock.getDelta();
                
                scene.children.forEach(c => {{
                    if(c.geometry && c.geometry.type === 'SphereGeometry') {{
                        c.rotation.y += 0.005;
                    }}
                }});
                particles.rotation.y += 0.0002;

                const speed = keyState.shift ? 150.0 : 40.0;
                
                velocity.x -= velocity.x * 5.0 * delta;
                velocity.z -= velocity.z * 5.0 * delta;
                
                direction.z = Number(keyState.w) - Number(keyState.s);
                direction.x = Number(keyState.d) - Number(keyState.a);
                direction.normalize();

                if (keyState.w || keyState.s) velocity.z -= direction.z * speed * delta;
                if (keyState.a || keyState.d) velocity.x -= direction.x * speed * delta;

                camera.translateX(velocity.x);
                camera.translateZ(velocity.z);

                // Proximity check
                let closeNode = null;
                let min_d = 25; 
                
                for(let i=0; i<interactables.length; i++) {{
                    const d = camera.position.distanceTo(interactables[i].position);
                    if(d < min_d) {{ min_d = d; closeNode = interactables[i]; }}
                }}
                
                if (closeNode) {{
                    if(lastHovered !== closeNode) {{
                        lastHovered = closeNode;
                        infoTitle.innerText = closeNode.userData.title || 'NODE';
                        infoDesc.innerText = closeNode.userData.content || '';
                        infoPanel.style.display = 'block';
                        infoPanel.style.opacity = '1';
                        closeNode.scale.set(1.5, 1.5, 1.5);
                        crosshair.style.transform = 'translate(-50%, -50%) scale(1.5)';
                        crosshair.style.borderColor = '#fff';
                    }}
                }} else {{
                    if(lastHovered) {{
                        lastHovered.scale.set(1, 1, 1);
                        lastHovered = null;
                        infoPanel.style.opacity = '0';
                        setTimeout(() => {{ if(!lastHovered) infoPanel.style.display = 'none'; }}, 300);
                        crosshair.style.transform = 'translate(-50%, -50%) scale(1)';
                        crosshair.style.borderColor = 'rgba(0, 255, 255, 0.3)';
                    }}
                }}

                renderer.render(scene, camera);
            }}

            animate();
        </script>
    </body>
    </html>
    """
    
    components.html(html_string, height=700, scrolling=False)

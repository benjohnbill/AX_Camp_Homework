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
                bottom: 80px;
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
            #cta-layer {{
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                display: flex;
                gap: 15px;
                pointer-events: auto;
            }}
            .btn-3d {{
                padding: 10px 24px;
                background: rgba(0, 255, 255, 0.15);
                border: 1px solid #0ff;
                color: #0ff;
                cursor: pointer;
                font-family: inherit;
                text-transform: uppercase;
                letter-spacing: 1px;
                border-radius: 4px;
                transition: all 0.2s;
            }}
            .btn-3d:hover {{
                background: rgba(0, 255, 255, 0.3);
                box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
            }}
            .btn-secondary {{
                background: rgba(255, 255, 255, 0.05);
                border-color: rgba(255, 255, 255, 0.3);
                color: rgba(255, 255, 255, 0.6);
            }}
            #legend {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(0,0,0,0.5);
                padding: 10px;
                border: 1px solid rgba(0,255,255,0.2);
                font-size: 10px;
                color: #fff;
            }}
            .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
            .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div class="hud-text">
            S Y S T E M : A N T I G R A V I T Y<br>
            W E E K L Y   R E P L A Y   O N L I N E<br>
            <span style="color:#0ff;">SCOPE: LAST 7 DAYS</span>
        </div>
        <div id="legend">
            <div class="legend-item"><div class="dot" style="background:#00ffff;"></div> Tier 1: Completed</div>
            <div class="legend-item"><div class="dot" style="background:#ff3333;"></div> Tier 2: Interrupted</div>
            <div class="legend-item"><div class="dot" style="background:#ffcc00;"></div> Tier 3: Evidence</div>
        </div>
        <div id="cta-layer">
            <button class="btn-3d" onclick="window.parent.postMessage({{type:'REPLAY_CLOSE'}}, '*')">Close Replay</button>
            <button class="btn-3d btn-secondary" onclick="window.parent.postMessage({{type:'REPLAY_SKIP'}}, '*')">Skip</button>
        </div>
        <div class="controls-hint">
            [W,A,S,D] NAVIGATE<br>
            [MOUSE DRAG] ROTATE<br>
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
            scene.fog = new THREE.FogExp2(0x0a0a1a, 0.003); 
            
            const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.1, 2000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);
            
            const ambient = new THREE.AmbientLight(0x222233);
            scene.add(ambient);
            const pointLight = new THREE.PointLight(0xffffff, 1, 1000);
            scene.add(pointLight);

            const interactables = [];
            
            if (coresData && coresData.length > 0) {{
                const coreGeo = new THREE.SphereGeometry(25, 32, 32);
                const coreMat = new THREE.MeshBasicMaterial({{ color: 0xe94560, wireframe: true, transparent: true, opacity: 0.8 }});
                const coreMesh = new THREE.Mesh(coreGeo, coreMat);
                coreMesh.userData = {{ isCore: true, content: coresData[0].content, title: "CORE CONSTITUTION" }};
                scene.add(coreMesh);
                interactables.push(coreMesh);
            }}

            const nodeGeo = new THREE.SphereGeometry(2, 16, 16);
            
            logsData.forEach((log) => {{
                if (!log.content || log.content.length < 5) return;
                const metaType = (log.meta_type || '').toLowerCase();

                const radius = 60 + Math.random() * 400;
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos((Math.random() * 2) - 1);
                
                const x = radius * Math.sin(phi) * Math.cos(theta);
                const y = radius * Math.sin(phi) * Math.sin(theta);
                const z = radius * Math.cos(phi);
                
                // Tier Logic
                let tier = 3; 
                let color = 0xffcc00; // Tier 3: Supporting Evidence / Default
                let typeText = "SUPPORTING EVIDENCE";

                if (metaType === 'session_completed') {{
                    tier = 1; color = 0x00ffff; typeText = "COMPLETED SESSION";
                }} else if (metaType === 'session_interrupted' || metaType === 'violation') {{
                    tier = 2; color = 0xff3333; typeText = "INTERRUPTED / VOID";
                }} else if (metaType === 'supporting_evidence') {{
                    tier = 3; color = 0xffcc00; typeText = "CURATED EVIDENCE";
                }}

                const sizeScale = tier === 1 ? 1.5 : (tier === 2 ? 1.2 : 0.8);
                const mat = new THREE.MeshPhongMaterial({{ 
                    color: color, 
                    emissive: color,
                    emissiveIntensity: 0.5,
                    transparent: true, 
                    opacity: 0.9 
                }});
                const mesh = new THREE.Mesh(nodeGeo, mat);
                mesh.scale.set(sizeScale, sizeScale, sizeScale);
                mesh.position.set(x, y, z);
                
                mesh.userData = {{ 
                    content: log.content, 
                    title: `[${{typeText}}] - ${{log.created_at ? log.created_at.substring(0,10) : ''}}`
                }};
                scene.add(mesh);
                interactables.push(mesh);
            }});

            const pGeo = new THREE.BufferGeometry();
            const pCount = 2000;
            const posArray = new Float32Array(pCount * 3);
            for(let i=0; i<pCount*3; i++) {{
                posArray[i] = (Math.random() - 0.5) * 1500;
            }}
            pGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
            const pMat = new THREE.PointsMaterial({{ size: 0.5, color: 0x44aaff, transparent: true, opacity: 0.3 }});
            const particles = new THREE.Points(pGeo, pMat);
            scene.add(particles);

            camera.position.z = 350;
            camera.lookAt(0,0,0);

            const keyState = {{ w:false, a:false, s:false, d:false }};
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
                
                const speed = 60.0;
                velocity.x -= velocity.x * 5.0 * delta;
                velocity.z -= velocity.z * 5.0 * delta;
                direction.z = Number(keyState.w) - Number(keyState.s);
                direction.x = Number(keyState.d) - Number(keyState.a);
                direction.normalize();
                if (keyState.w || keyState.s) velocity.z -= direction.z * speed * delta;
                if (keyState.a || keyState.d) velocity.x -= direction.x * speed * delta;
                camera.translateX(velocity.x);
                camera.translateZ(velocity.z);

                let closeNode = null;
                let min_d = 30; 
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
                        crosshair.style.transform = 'translate(-50%, -50%) scale(1.5)';
                        crosshair.style.borderColor = '#fff';
                    }}
                }} else {{
                    if(lastHovered) {{
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

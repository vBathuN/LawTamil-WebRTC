import * as THREE from 'three/webgpu';
import { pass, mrt, output, normalView, diffuseColor, velocity, add, vec4, directionToColor, colorToDirection, sample } from 'three/tsl';
import { ssgi } from 'three/addons/tsl/display/SSGINode.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

// MediaPipe (face tracking)
import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';

// ── Config ──

const BOX_HEIGHT = 6;
const BOX_DEPTH = 8;
const WALL_THICKNESS = 0.5;
const CAM_FOV = 45;
const FACE_FILL = 0.62; // fraction of box height the face should occupy
const LIGHT_DEPTH = BOX_DEPTH / 2 + 2; // point-light plane, forward toward the camera

function getBoxWidth() {

	const aspect = window.innerWidth / window.innerHeight;
	const vFov = THREE.MathUtils.degToRad( CAM_FOV / 2 );
	// Width that fills the viewport at the front face distance
	const dist = ( BOX_HEIGHT / 2 ) / Math.tan( vFov );
	return Math.tan( vFov ) * aspect * dist * 2;

}

// MediaPipe blendshape name → facecap morph target name
const blendshapesMap = {
	// '_neutral': '',
	'browDownLeft': 'browDown_L',
	'browDownRight': 'browDown_R',
	'browInnerUp': 'browInnerUp',
	'browOuterUpLeft': 'browOuterUp_L',
	'browOuterUpRight': 'browOuterUp_R',
	'cheekPuff': 'cheekPuff',
	'cheekSquintLeft': 'cheekSquint_L',
	'cheekSquintRight': 'cheekSquint_R',
	'eyeBlinkLeft': 'eyeBlink_L',
	'eyeBlinkRight': 'eyeBlink_R',
	'eyeLookDownLeft': 'eyeLookDown_L',
	'eyeLookDownRight': 'eyeLookDown_R',
	'eyeLookInLeft': 'eyeLookIn_L',
	'eyeLookInRight': 'eyeLookIn_R',
	'eyeLookOutLeft': 'eyeLookOut_L',
	'eyeLookOutRight': 'eyeLookOut_R',
	'eyeLookUpLeft': 'eyeLookUp_L',
	'eyeLookUpRight': 'eyeLookUp_R',
	'eyeSquintLeft': 'eyeSquint_L',
	'eyeSquintRight': 'eyeSquint_R',
	'eyeWideLeft': 'eyeWide_L',
	'eyeWideRight': 'eyeWide_R',
	'jawForward': 'jawForward',
	'jawLeft': 'jawLeft',
	'jawOpen': 'jawOpen',
	'jawRight': 'jawRight',
	'mouthClose': 'mouthClose',
	'mouthDimpleLeft': 'mouthDimple_L',
	'mouthDimpleRight': 'mouthDimple_R',
	'mouthFrownLeft': 'mouthFrown_L',
	'mouthFrownRight': 'mouthFrown_R',
	'mouthFunnel': 'mouthFunnel',
	'mouthLeft': 'mouthLeft',
	'mouthLowerDownLeft': 'mouthLowerDown_L',
	'mouthLowerDownRight': 'mouthLowerDown_R',
	'mouthPressLeft': 'mouthPress_L',
	'mouthPressRight': 'mouthPress_R',
	'mouthPucker': 'mouthPucker',
	'mouthRight': 'mouthRight',
	'mouthRollLower': 'mouthRollLower',
	'mouthRollUpper': 'mouthRollUpper',
	'mouthShrugLower': 'mouthShrugLower',
	'mouthShrugUpper': 'mouthShrugUpper',
	'mouthSmileLeft': 'mouthSmile_L',
	'mouthSmileRight': 'mouthSmile_R',
	'mouthStretchLeft': 'mouthStretch_L',
	'mouthStretchRight': 'mouthStretch_R',
	'mouthUpperUpLeft': 'mouthUpperUp_L',
	'mouthUpperUpRight': 'mouthUpperUp_R',
	'noseSneerLeft': 'noseSneer_L',
	'noseSneerRight': 'noseSneer_R',
	// '': 'tongueOut'
};

// ── Three.js ──

let camera, scene, renderer, renderPipeline;
let raycaster, pointer;
let mouseLight;

// ── Cornell box ──

let wallMeshes = [];
let boxSize = { w: 8, h: BOX_HEIGHT, d: BOX_DEPTH };

// ── Face ──

let face, eyeL, eyeR, faceTransform;
let faceRoot, faceBaseSize, faceBaseCenter, faceBaseScale;
let faceLandmarker, video;
const eyeRotationLimit = THREE.MathUtils.degToRad( 30 );
const transform = new THREE.Object3D();

// ── Mouse light easing ──

let mouseLightTarget = new THREE.Vector3( 0, BOX_HEIGHT / 2, LIGHT_DEPTH );
const EASE_SPEED = 8;

// ── Init ──

init();

async function init() {

	// Camera

	camera = new THREE.PerspectiveCamera( CAM_FOV, window.innerWidth / window.innerHeight, 0.1, 100 );

	// Scene

	scene = new THREE.Scene();

	// Renderer

	renderer = new THREE.WebGPURenderer( { antialias: false } );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 0.5;
	renderer.shadowMap.enabled = true;
	document.body.appendChild( renderer.domElement );

	// WebGPU needs the device ready before KTX2Loader.detectSupport()
	await renderer.init();

	// Post-processing: SSGI

	renderPipeline = new THREE.RenderPipeline( renderer );

	const scenePass = pass( scene, camera );
	scenePass.setMRT( mrt( {
		output: output,
		diffuseColor: diffuseColor,
		normal: directionToColor( normalView ),
		velocity: velocity,
	} ) );

	const scenePassColor = scenePass.getTextureNode( 'output' );
	const scenePassDiffuse = scenePass.getTextureNode( 'diffuseColor' );
	const scenePassDepth = scenePass.getTextureNode( 'depth' );
	const scenePassNormal = scenePass.getTextureNode( 'normal' );
	const scenePassVelocity = scenePass.getTextureNode( 'velocity' );

	const diffuseTexture = scenePass.getTexture( 'diffuseColor' );
	diffuseTexture.type = THREE.UnsignedByteType;
	const normalTexture = scenePass.getTexture( 'normal' );
	normalTexture.type = THREE.UnsignedByteType;

	const sceneNormal = sample( ( uv ) => {

		return colorToDirection( scenePassNormal.sample( uv ) );

	} );

	const giPass = ssgi( scenePassColor, scenePassDepth, sceneNormal, camera );
	giPass.sliceCount.value = 2;
	giPass.stepCount.value = 8;
	giPass.giIntensity.value = 18;
	giPass.aoIntensity.value = 0.55;

	const ao = giPass.getAONode();
	const gi = giPass.getGINode();

	const compositePass = vec4(
		add( scenePassColor.rgb.mul( ao.r ), scenePassDiffuse.rgb.mul( gi.rgb ) ),
		scenePassColor.a
	);

	const traaPass = traa( compositePass, scenePassDepth, scenePassVelocity, camera );
	renderPipeline.outputNode = traaPass;

	// Lights

	mouseLight = new THREE.PointLight( 0xffffff, 80 );
	mouseLight.position.set( 0, BOX_HEIGHT / 2, LIGHT_DEPTH );
	mouseLight.castShadow = true;
	mouseLight.shadow.mapSize.set( 1024, 1024 );
	mouseLight.shadow.radius = 20;
	scene.add( mouseLight );

	// Raycaster (used to steer the light)

	raycaster = new THREE.Raycaster();
	pointer = new THREE.Vector2();

	// Build the box + fit the camera

	rebuildWalls();

	// Render the box right away while the face + tracking load

	renderer.setAnimationLoop( animate );

	// Load the face model, then start webcam tracking

	loadFace();
	await initFaceTracking();

	// Events

	renderer.domElement.addEventListener( 'pointermove', onPointerMove );
	window.addEventListener( 'resize', onResize );

}

// ── Cornell box ──

function rebuildWalls() {

	for ( const mesh of wallMeshes ) {

		scene.remove( mesh );
		mesh.geometry.dispose();

	}

	wallMeshes = [];

	// Box width depends on the viewport aspect ratio
	boxSize.w = getBoxWidth();

	fitCameraToBox();

	createBox();

	// Re-fit the face for the new aspect ratio (no-op until it has loaded)
	fitFace();

}

function createBox() {

	const hw = boxSize.w / 2;
	const hh = boxSize.h / 2;
	const hd = boxSize.d / 2;
	const t = WALL_THICKNESS;

	const whiteMaterial = new THREE.MeshPhysicalMaterial( {
		color: 0xeeeeee,
		roughness: 0.7,
		metalness: 0.0,
	} );

	const redMaterial = new THREE.MeshPhysicalMaterial( {
		color: 0xff2222,
		roughness: 0.7,
		metalness: 0.0,
	} );

	const greenMaterial = new THREE.MeshPhysicalMaterial( {
		color: 0x22ff22,
		roughness: 0.7,
		metalness: 0.0,
	} );

	const walls = [
		// floor
		{ size: [ boxSize.w, t, boxSize.d ], pos: [ 0, - t / 2, 0 ], mat: whiteMaterial },
		// ceiling
		{ size: [ boxSize.w, t, boxSize.d ], pos: [ 0, boxSize.h + t / 2, 0 ], mat: whiteMaterial },
		// back
		{ size: [ boxSize.w, boxSize.h, t ], pos: [ 0, hh, - hd - t / 2 ], mat: whiteMaterial },
		// left
		{ size: [ t, boxSize.h, boxSize.d ], pos: [ - hw - t / 2, hh, 0 ], mat: redMaterial },
		// right
		{ size: [ t, boxSize.h, boxSize.d ], pos: [ hw + t / 2, hh, 0 ], mat: greenMaterial },
	];

	for ( const w of walls ) {

		const geo = new THREE.BoxGeometry( w.size[ 0 ], w.size[ 1 ], w.size[ 2 ] );
		const mesh = new THREE.Mesh( geo, w.mat );
		mesh.position.set( ...w.pos );
		mesh.receiveShadow = true;
		scene.add( mesh );
		wallMeshes.push( mesh );

	}

}

// ── Face ──

function loadFace() {

	const ktx2Loader = new KTX2Loader()
		.setTranscoderPath( 'https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/libs/basis/' )
		.detectSupport( renderer );

	new GLTFLoader()
		.setKTX2Loader( ktx2Loader )
		.setMeshoptDecoder( MeshoptDecoder )
		.load( 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r184/examples/models/gltf/facecap.glb', ( gltf ) => {

			const mesh = gltf.scene.children[ 0 ];

			// Lit clay so the face is shaded by the box light + SSGI bounce
			const clayMaterial = new THREE.MeshStandardMaterial( {
				color: 0xdddddd,
				roughness: 0.6,
				metalness: 1.0,
				side: THREE.DoubleSide,
			} );

			const head = mesh.getObjectByName( 'mesh_2' );
			head.material = clayMaterial;
			head.castShadow = true;
			head.receiveShadow = true;

			const teeth = mesh.getObjectByName( 'mesh_3' );
			teeth.material = clayMaterial;
			teeth.receiveShadow = true;

			face = head;
			eyeL = mesh.getObjectByName( 'eyeLeft' );
			eyeR = mesh.getObjectByName( 'eyeRight' );
			faceTransform = mesh.getObjectByName( 'grp_transform' );

			// Mirror horizontally for a selfie-like feel (localized to the face,
			// so the box's red/green walls stay put). three.js fixes the winding.
			mesh.scale.x *= - 1;

			scene.add( mesh );

			// Measure the face at its base (mirrored) scale, then fit it to the box.
			// We keep the base measurements so we can re-fit on resize / orientation change.
			scene.updateMatrixWorld( true );

			const bbox = new THREE.Box3().setFromObject( mesh );
			faceRoot = mesh;
			faceBaseScale = mesh.scale.clone();
			faceBaseSize = bbox.getSize( new THREE.Vector3() );
			faceBaseCenter = bbox.getCenter( new THREE.Vector3() );

			fitFace();

		} );

}

function fitFace() {

	if ( ! faceRoot ) return;

	// Size the face relative to the smaller viewport dimension so it doesn't
	// dominate in portrait. Landscape (aspect >= 1) keeps the full size.
	const aspect = window.innerWidth / window.innerHeight;
	const aspectFill = Math.min( 1, aspect );

	const s = ( boxSize.h * FACE_FILL * 3 * aspectFill ) / faceBaseSize.y;

	faceRoot.scale.copy( faceBaseScale ).multiplyScalar( s );
	faceRoot.position.set(
		- faceBaseCenter.x * s,
		boxSize.h / 2 - faceBaseCenter.y * s,
		- faceBaseCenter.z * s
	);

}

// 🟢 மாற்றப்பட்ட WebRTC / VDO.Ninja Face Tracking பகுதி
async function initFaceTracking() {

	video = document.createElement( 'video' );
	video.muted = true;
	video.playsInline = true;
	video.crossOrigin = "anonymous"; // Cross-Origin தடையைத் தவிர்க்க

	const filesetResolver = await FilesetResolver.forVisionTasks(
		'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
	);

	faceLandmarker = await FaceLandmarker.createFromOptions( filesetResolver, {
		baseOptions: {
			modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
			delegate: 'GPU'
		},
		outputFaceBlendshapes: true,
		outputFacialTransformationMatrixes: true,
		runningMode: 'VIDEO',
		numFaces: 1
	} );

    // VDO.Ninja WebRTC Iframe-ஐ உருவாக்குதல்
    const roomUrl = "https://webrtc.lawtamil.in/?room=LTStrial2026&push=&wc&as&hh&ee&effect&label=&mobile&hidetranslate&api=true";
    
    const vdoIframe = document.createElement('iframe');
    vdoIframe.src = roomUrl;
    vdoIframe.style.display = 'none'; // திரையில் தெரியாமல் மறைக்க
    vdoIframe.allow = "camera; microphone; autoplay; display-capture";
    document.body.appendChild(vdoIframe);

    // Iframe லோட் ஆனதும் அதிலிருந்து Video Stream-ஐப் பெற முயற்சி
    vdoIframe.onload = () => {
        try {
            const iframeDoc = vdoIframe.contentDocument || vdoIframe.contentWindow.document;
            const iframeVideo = iframeDoc.querySelector('video');
            
            if (iframeVideo) {
                // Iframe-ல் உள்ள வீடியோவை MediaPipe வீடியோவுடன் இணைக்கிறோம்
                iframeVideo.onloadedmetadata = () => {
                    video.srcObject = iframeVideo.captureStream ? iframeVideo.captureStream() : (iframeVideo.mozCaptureStream ? iframeVideo.mozCaptureStream() : iframeVideo.srcObject);
                    video.play().catch(e => console.warn(e));
                };
                
                // ஏற்கனவே Iframe வீடியோ ஓடிக்கொண்டிருந்தால்
                if(iframeVideo.readyState >= 3) {
                    video.srcObject = iframeVideo.captureStream ? iframeVideo.captureStream() : (iframeVideo.mozCaptureStream ? iframeVideo.mozCaptureStream() : iframeVideo.srcObject);
                    video.play().catch(e => console.warn(e));
                }
            }
        } catch (err) {
            console.warn("Cross-Origin தடையால் நேரடியாக எடுக்க முடியவில்லை. Message API-க்கு மாறுகிறது.", err);
        }
    };

    // Fallback: Message API மூலம் Stream-ஐப் பெற
    window.addEventListener('message', (event) => {
        if (event.data && event.data.type === 'remoteStream' && event.data.stream) {
             video.srcObject = event.data.stream;
             video.play().catch(e => console.error("Video play error:", e));
        }
    });

}

// ── Light interaction ──

function onPointerMove( event ) {

	pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
	pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

	raycaster.setFromCamera( pointer, camera );

	// Steer the point light across the front face plane of the box
	const frontPlane = new THREE.Plane( new THREE.Vector3( 0, 0, 1 ), - LIGHT_DEPTH );
	const hit = new THREE.Vector3();
	if ( raycaster.ray.intersectPlane( frontPlane, hit ) ) {

		mouseLightTarget.copy( hit );

	}

}

// ── Camera fitting ──

function fitCameraToBox() {

	const aspect = window.innerWidth / window.innerHeight;
	const vFov = THREE.MathUtils.degToRad( CAM_FOV / 2 );

	const dist = ( boxSize.h / 2 ) / Math.tan( vFov );

	camera.aspect = aspect;
	camera.position.set( 0, boxSize.h / 2, dist + boxSize.d / 2 );
	camera.lookAt( 0, boxSize.h / 2, 0 );
	camera.updateProjectionMatrix();

}

// ── Resize ──

function onResize() {

	renderer.setSize( window.innerWidth, window.innerHeight );
	rebuildWalls();

}

// ── Animate ──

const timer = new THREE.Timer();

function animate() {

	timer.update();
	const dt = Math.min( timer.getDelta(), 1 / 30 );

	// Ease the point light toward the cursor
	const easeFactor = 1 - Math.exp( - EASE_SPEED * dt );
	mouseLight.position.lerp( mouseLightTarget, easeFactor );

	// Drive the face from the webcam
	if ( faceLandmarker && face && video && video.readyState >= HTMLMediaElement.HAVE_METADATA ) {

		const results = faceLandmarker.detectForVideo( video, performance.now() );

		// Head pose
		if ( results.facialTransformationMatrixes.length > 0 && faceTransform ) {

			const matrix = results.facialTransformationMatrixes[ 0 ].data;

			transform.matrix.fromArray( matrix );
			transform.matrix.decompose( transform.position, transform.quaternion, transform.scale );

			faceTransform.position.x = transform.position.x;
			faceTransform.position.y = transform.position.z + 40;
			faceTransform.position.z = - transform.position.y;

			faceTransform.rotation.x = transform.rotation.x;
			faceTransform.rotation.y = transform.rotation.z;
			faceTransform.rotation.z = - transform.rotation.y;

		}

		// Blendshapes (expressions + eye gaze)
		if ( results.faceBlendshapes.length > 0 ) {

			const faceBlendshapes = results.faceBlendshapes[ 0 ].categories;

			// Eyes have no morph targets, so map their blendshape scores to rotation
			const eyeScore = {
				leftHorizontal: 0,
				rightHorizontal: 0,
				leftVertical: 0,
				rightVertical: 0,
			};

			for ( const blendshape of faceBlendshapes ) {

				const categoryName = blendshape.categoryName;
				const score = blendshape.score;

				const index = face.morphTargetDictionary[ blendshapesMap[ categoryName ] ];

				if ( index !== undefined ) {

					face.morphTargetInfluences[ index ] = score;

				}

				// Two blendshapes per axis (up/down, in/out): add one, subtract the
				// other for a final score in the -1 to 1 range
				switch ( categoryName ) {

					case 'eyeLookInLeft':
						eyeScore.leftHorizontal += score;
						break;
					case 'eyeLookOutLeft':
						eyeScore.leftHorizontal -= score;
						break;
					case 'eyeLookInRight':
						eyeScore.rightHorizontal -= score;
						break;
					case 'eyeLookOutRight':
						eyeScore.rightHorizontal += score;
						break;
					case 'eyeLookUpLeft':
						eyeScore.leftVertical -= score;
						break;
					case 'eyeLookDownLeft':
						eyeScore.leftVertical += score;
						break;
					case 'eyeLookUpRight':
						eyeScore.rightVertical -= score;
						break;
					case 'eyeLookDownRight':
						eyeScore.rightVertical += score;
						break;

				}

			}

			eyeL.rotation.z = eyeScore.leftHorizontal * eyeRotationLimit;
			eyeR.rotation.z = eyeScore.rightHorizontal * eyeRotationLimit;
			eyeL.rotation.x = eyeScore.leftVertical * eyeRotationLimit;
			eyeR.rotation.x = eyeScore.rightVertical * eyeRotationLimit;

		}

	}

	renderPipeline.render();

}
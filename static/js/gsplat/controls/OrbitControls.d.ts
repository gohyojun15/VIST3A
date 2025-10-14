import { Camera } from '../cameras/Camera';
import { Vector3 } from '../math/Vector3';
declare class OrbitControls {
    minAngle: number;
    maxAngle: number;
    minZoom: number;
    maxZoom: number;
    orbitSpeed: number;
    panSpeed: number;
    zoomSpeed: number;
    dampening: number;
    setCameraTarget: (newTarget: Vector3) => void;
    update: () => void;
    dispose: () => void;
    constructor(camera: Camera, canvas: HTMLElement, alpha?: number, beta?: number, radius?: number, enableKeyboardControls?: boolean, inputTarget?: Vector3);
}
export { OrbitControls };

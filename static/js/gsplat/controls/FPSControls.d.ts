import { Camera } from '../cameras/Camera';
declare class FPSControls {
    moveSpeed: number;
    lookSpeed: number;
    dampening: number;
    update: () => void;
    dispose: () => void;
    constructor(camera: Camera, canvas: HTMLCanvasElement);
}
export { FPSControls };

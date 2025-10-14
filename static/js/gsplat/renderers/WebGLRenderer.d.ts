import { Scene } from '../core/Scene';
import { Camera } from '../cameras/Camera';
import { Color32 } from '../math/Color32';
import { ShaderProgram } from './webgl/programs/ShaderProgram';
import { RenderProgram } from './webgl/programs/RenderProgram';
import { ShaderPass } from './webgl/passes/ShaderPass';
export declare class WebGLRenderer {
    private _canvas;
    private _gl;
    private _backgroundColor;
    private _renderProgram;
    addProgram: (program: ShaderProgram) => void;
    removeProgram: (program: ShaderProgram) => void;
    resize: () => void;
    setSize: (width: number, height: number) => void;
    render: (scene: Scene, camera: Camera) => void;
    dispose: () => void;
    constructor(optionalCanvas?: HTMLCanvasElement | null, optionalRenderPasses?: ShaderPass[] | null);
    get canvas(): HTMLCanvasElement;
    get gl(): WebGL2RenderingContext;
    get renderProgram(): RenderProgram;
    get backgroundColor(): Color32;
    set backgroundColor(value: Color32);
}

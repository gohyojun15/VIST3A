import { SplatvData } from '../../../splats/SplatvData';
import { WebGLRenderer } from '../../WebGLRenderer';
import { ShaderPass } from '../passes/ShaderPass';
import { ShaderProgram } from './ShaderProgram';
declare class VideoRenderProgram extends ShaderProgram {
    private _renderData;
    private _depthIndex;
    private _splatTexture;
    protected _initialize: () => void;
    protected _resize: () => void;
    protected _render: () => void;
    protected _dispose: () => void;
    constructor(renderer: WebGLRenderer, passes?: ShaderPass[]);
    get renderData(): SplatvData | null;
    protected _getVertexSource(): string;
    protected _getFragmentSource(): string;
}
export { VideoRenderProgram };

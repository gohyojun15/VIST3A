import { ShaderProgram } from '../programs/ShaderProgram';
import { ShaderPass } from './ShaderPass';
declare class FadeInPass implements ShaderPass {
    initialize: (program: ShaderProgram) => void;
    render: () => void;
    constructor(speed?: number);
    dispose(): void;
}
export { FadeInPass };

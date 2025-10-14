import { ShaderProgram } from '../programs/ShaderProgram';
declare class ShaderPass {
    initialize(program: ShaderProgram): void;
    render(): void;
    dispose(): void;
}
export { ShaderPass };

declare class EventDispatcher {
    addEventListener: (type: string, listener: (event: Event) => void) => void;
    removeEventListener: (type: string, listener: (event: Event) => void) => void;
    hasEventListener: (type: string, listener: (event: Event) => void) => boolean;
    dispatchEvent: (event: Event) => void;
    constructor();
}
export { EventDispatcher };

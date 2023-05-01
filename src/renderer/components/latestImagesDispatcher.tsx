import { useState, useEffect, useReducer } from "react";
import { useRecoilValue } from "recoil";
import { latestImageState } from "../atoms/atoms";

interface ReducerState {
    selectedIndex: number;
}
  
export interface ReducerAction {
    type: "arrowLeft" | "arrowRight" | "select" | "intermediate";
    payload?: number;
}

export const useKeyPress = (targetKey: string, useAltKey = false) => {
    const [keyPressed, setKeyPressed] = useState(false);

    useEffect(() => {
        const handler = (isKeyDown: boolean) => ({ key, altKey } : { key: string, altKey: boolean }) => {
            if (key === targetKey && altKey === useAltKey) {
                console.log(`key: ${key}, altKey: ${altKey}, keydown: ${isKeyDown}`);
                setKeyPressed(isKeyDown);
            }
        };

        const keydownHandler = handler(true);
        const keyupHandler = handler(false);

        window.addEventListener('keydown', keydownHandler);
        window.addEventListener('keyup', keyupHandler);

        return () => {
            window.removeEventListener('keydown', keydownHandler);
            window.removeEventListener('keyup', keyupHandler);
        };
    }, [targetKey, useAltKey]);

    return keyPressed;
}

export const latestImagesDispatcher = () => {
    const latestImages = useRecoilValue(latestImageState);

    const reducer = (state: ReducerState, action: ReducerAction): ReducerState => {
        switch (action.type) {
        case 'arrowLeft':
            console.log('Arrow Left');
            return {
                selectedIndex:
              state.selectedIndex !== 0
                  ? state.selectedIndex - 1
                  : latestImages.length - 1
            };
        case 'arrowRight':
            console.log('Arrow Right');
            return {
                selectedIndex:
              state.selectedIndex !== latestImages.length - 1
                  ? state.selectedIndex + 1
                  : 0
            };
        case 'select':
            console.log('Select');
            return { selectedIndex: action.payload ?? -1 };
        case 'intermediate':
            console.log('intermediate');
            return { selectedIndex: -1 };
        default:
            throw new Error();
        }
    };

    return useReducer(reducer, { selectedIndex: 0 });
}


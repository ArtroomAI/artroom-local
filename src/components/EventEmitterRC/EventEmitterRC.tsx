import * as React from 'react';
import { createContext } from 'react';
import { EventEmitter } from '../../helpers/EventEmitter';

type PropsType = {
	children?: JSX.Element;
};

export const EventEmitterRCContext = createContext<EventEmitter<any>>(
	null as any,
);

export const EventEmitterRC: React.FC<PropsType> = props => {
	return (
		<EventEmitterRCContext.Provider value={new EventEmitter()}>
			{props.children}
		</EventEmitterRCContext.Provider>
	);
};

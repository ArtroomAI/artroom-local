import { Box } from '@chakra-ui/react';
import React, { useContext } from 'react';
import { ContextMenuContext } from './ContextMenu';

// Type Props = {};

export default function ContextMenuTrigger ({ children }: { children: React.ReactNode }) {
    const { openMenu, setPosition } = useContext(ContextMenuContext);
    return (
        <Box
            onContextMenu={(event) => {
                event.preventDefault();
                setPosition({ x: event.clientX,
                    y: event.clientY });
                openMenu();
            }}
        >
            {children}
        </Box>
    );
}

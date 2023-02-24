import React, { useContext, useState } from 'react';
import { Button, ThemeTypings } from '@chakra-ui/react';
import { ContextMenuContext } from './ContextMenu';

export default function ContextMenuItem ({
    onClick,
    colorScheme,
    disabled = false
} : {
    children: React.ReactNode;
    onClick: (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => void;
    colorScheme: ThemeTypings["colorSchemes"];
    disabled: boolean;
}) {
    const [variant, setVariant] = useState('ghost');
    const { closeMenu } = useContext(ContextMenuContext);
    return (
        <Button
            borderRadius={0}
            colorScheme={colorScheme}
            disabled={disabled}
            justifyContent="left"
            onClick={(e) => {
                e.preventDefault();
                onClick(e);
                closeMenu();
            }}
            onMouseOut={() => setVariant('ghost')}
            onMouseOver={() => setVariant('solid')}
            overflow="hidden"
            size="sm"
            textOverflow="ellipsis"
            variant={variant}
            w="100%"
        >
        </Button>
    );
}

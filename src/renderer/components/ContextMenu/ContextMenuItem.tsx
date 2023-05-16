import React, { useContext, useState } from 'react';
import { Button } from '@chakra-ui/react';
import { ContextMenuContext } from './ContextMenu';

export default function ContextMenuItem ({
    children,
    onClick,
    disabled = false
} : {
    children: React.ReactNode;
    onClick: (ev: React.MouseEvent<HTMLButtonElement, MouseEvent>) => void;
    disabled?: boolean;
}) {
    const [variant, setVariant] = useState('ghost');
    const { closeMenu } = useContext(ContextMenuContext);
    return (
        <Button
            borderRadius={0}
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
            {children}
        </Button>
    );
}

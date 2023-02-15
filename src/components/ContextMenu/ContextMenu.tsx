import { useDisclosure } from '@chakra-ui/react'
import React, { useRef, useState } from 'react'

export const ContextMenuContext = React.createContext({
  isOpen: false,
  closeMenu: () => {},
  openMenu: () => {},
  menuRef: undefined,
  position: { x: 0, y: 0 },
  setPosition: ({ x, y }: { x: number; y: number }) => {}
})

export default function ContextMenu({
  children
}: {
  children: React.ReactNode
}) {
  const { isOpen, onClose: closeMenu, onOpen: openMenu } = useDisclosure()
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const menuRef = useRef(null)
  return (
    <ContextMenuContext.Provider
      value={{
        isOpen,
        closeMenu,
        openMenu,
        menuRef,
        position,
        setPosition
      }}
    >
      {children}
    </ContextMenuContext.Provider>
  )
}

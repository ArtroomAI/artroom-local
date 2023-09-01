import { useDisclosure } from '@chakra-ui/react'
import React, { useRef, useState } from 'react'

type ContextMenuContextType = {
  isOpen: boolean
  closeMenu: () => void
  openMenu: () => void
  menuRef: React.MutableRefObject<HTMLDivElement | null>
  position: { x: number; y: number }
  setPosition: ({ x, y }: { x: number; y: number }) => void
}

export const ContextMenuContext = React.createContext<ContextMenuContextType>({
  isOpen: false,
  closeMenu: () => {},
  openMenu: () => {},
  menuRef: { current: null },
  position: { x: 0, y: 0 },
  setPosition: ({ x, y }) => {},
})

export default function ContextMenu({ children }: { children: React.ReactNode }) {
  const { isOpen, onClose: closeMenu, onOpen: openMenu } = useDisclosure()
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const menuRef = useRef<HTMLDivElement | null>(null)
  return (
    <ContextMenuContext.Provider
      value={{
        isOpen,
        closeMenu,
        openMenu,
        menuRef,
        position,
        setPosition,
      }}
    >
      {children}
    </ContextMenuContext.Provider>
  )
}

import React, { useEffect, useState } from 'react'
import { Box, Progress, Text } from '@chakra-ui/react'
import { ProgressInfo } from 'electron-updater'

export const UpdateProgressBar: React.FC = () => {
  const [info, setInfo] = useState<ProgressInfo>({
    percent: 0,
    total: 0,
    transferred: 0,
    bytesPerSecond: 0,
    delta: 0,
  })

  useEffect(() => {
    const handlerDiscard = window.api.downloadProgress((_, _info) => {
      setInfo(_info)
    })

    return () => {
      handlerDiscard()
    }
  }, [])

  if (info.percent && info.percent < 100) {
    return (
      <Box position="fixed" bottom="0" left="0" right="0" textAlign="center">
        <Progress height="26px" colorScheme="purple" value={info.percent} />
        <Text
          position="absolute"
          bottom="0"
          left="0"
          right="0"
          textAlign="center"
          textShadow={'-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000'}
        >
          Downloading update: {(info.transferred / 1000 / 1000).toFixed(2)}MB /{' '}
          {(info.total / 1000 / 1000).toFixed(2)}MB -{' '}
          {(info.bytesPerSecond / 1000 / 1000).toFixed(2)}MB/s
        </Text>
      </Box>
    )
  }

  return <></>
}

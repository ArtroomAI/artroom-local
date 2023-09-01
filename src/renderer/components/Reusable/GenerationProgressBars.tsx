import React, { useCallback, useContext, useEffect, useState } from 'react'
import { SocketContext, SocketOnEvents } from '../../socket'
import { Box, Progress } from '@chakra-ui/react'

// parses mm:ssit/s or s/it into number
const parseIterations = (inputStr: string) => {
  const match = inputStr.match(/([\d.]+)(it\/s|s\/it)/)
  if (match) {
    const value = parseFloat(match[1])
    const unit = match[2]

    // Converting the value to iterations per second
    let iterationsPerSec = 0
    if (unit === 'it/s') {
      iterationsPerSec = value
    } else if (unit === 's/it') {
      iterationsPerSec = 1 / value
    }
    return iterationsPerSec
  }
  return 0
}

const parseToTime = (seconds: number) => {
  seconds = Math.round(seconds)
  const s = seconds % 60
  const m = Math.floor(seconds / 60)

  return `${m}:${s > 10 ? s : `0${s}`}`
}

const getPercent = (a: number, b: number) => {
  return a === b ? 100 : (100 * a) / b
}

export const GenerationProgressBars = () => {
  const socket = useContext(SocketContext)

  const [progress, setProgress] = useState({
    current: -1,
    batch: -1,
    timeSpent: '',
    currentEta: '',
    batchEta: '',
  })

  const handleGetProgress: SocketOnEvents['get_progress'] = useCallback((data) => {
    console.log(data)
    const totalStepsDone = data.current_num * data.total_steps + data.current_step
    const totalSteps = data.total_steps * data.total_num

    const current = getPercent(data.current_step, data.total_steps)
    const batch = getPercent(totalStepsDone, totalSteps)

    const perSecond = parseIterations(data.iterations_per_sec)
    const timeSpent = data.time_spent
    const currentEta = data.eta
    const batchEta = parseToTime((totalSteps - totalStepsDone) / perSecond)

    console.log(data, { current, batch, timeSpent, currentEta, batchEta })
    setProgress({ current, batch, timeSpent, currentEta, batchEta })
  }, [])

  // on socket message
  useEffect(() => {
    socket.on('get_progress', handleGetProgress)

    return () => {
      socket.off('get_progress', handleGetProgress)
    }
  }, [socket, handleGetProgress])

  if (progress.current >= 0 && progress.current < 100) {
    return (
      <>
        <Progress alignContent="left" hasStripe width="100%" value={progress.batch} />
        <Progress alignContent="left" hasStripe width="100%" value={progress.current} />
        <Box>
          eta: {progress.currentEta} | batch eta: {progress.batchEta}
        </Box>
      </>
    )
  }

  return null
}

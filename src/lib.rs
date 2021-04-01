#![forbid(unsafe_code)]

use std::{ops::{Deref, DerefMut}};
use std::{sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard}};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Clone)]
struct InnerMarchingBuffer<T> {
    data: Arc<RwLock<Vec<T>>>,
    /// Number of entries that have been finished in the buffer.
    finished_len: Arc<AtomicUsize>,
    /// How many Readers exist.
    readers: Arc<AtomicUsize>,
    /// True if a Writer exists, false otherwise.
    has_writer: Arc<AtomicBool>,
    /// Offset into the data vec of where the writable section starts. Equivalently, the total amount of data that has
    /// been frozen for reading. This is reset to 0 once all Readers and Writers are dropped. This is updated whenever a WriterAccess is dropped.
    write_offset: Arc<AtomicUsize>,
}

impl<T> InnerMarchingBuffer<T> {
    fn check_reset(&self) {
        if let Ok(mut data) = self.data.try_write() {
            if self.readers.load(Ordering::SeqCst) == 0 && !self.has_writer.load(Ordering::SeqCst) {
                self.write_offset.store(0, Ordering::SeqCst);
                self.finished_len.store(0, Ordering::SeqCst);
                data.clear();
            }
        }
    }
}

impl<T> std::fmt::Debug for InnerMarchingBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.data.try_read() {
            Ok(data) => {
                f.debug_struct("InnerMarchingBuffer")
                    .field("data_len", &data.len())
                    .field("data_capacity", &data.capacity())
                    .field("finished_len", &self.finished_len.load(Ordering::SeqCst))
                    .field("readers", &self.readers.load(Ordering::SeqCst))
                    .field("has_writer", &self.has_writer.load(Ordering::SeqCst))
                    .field("write_offset", &self.write_offset.load(Ordering::SeqCst))
                    .finish()
            }
            Err(_) => {
                f.debug_struct("InnerMarchingBuffer")
                    .field("data_len", &"(locked)")
                    .field("data_capacity", &"(locked)")
                    .field("finished_len", &self.finished_len.load(Ordering::SeqCst))
                    .field("readers", &self.readers.load(Ordering::SeqCst))
                    .field("has_writer", &self.has_writer.load(Ordering::SeqCst))
                    .field("write_offset", &self.write_offset.load(Ordering::SeqCst))
                    .finish()
            }
        }
    }
}

#[derive(Clone)]
pub struct MarchingBuffer<T> {
    inner: Arc<InnerMarchingBuffer<T>>
}

impl<T> MarchingBuffer<T> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(InnerMarchingBuffer {
                data: Arc::new(RwLock::new(Vec::new())),
                finished_len: Arc::new(AtomicUsize::new(0)),
                readers: Arc::new(AtomicUsize::new(0)),
                has_writer: Arc::new(AtomicBool::new(false)),
                write_offset: Arc::new(AtomicUsize::new(0))
            })
        }
    }

    pub fn finished_len(&self) -> usize {
        self.inner.finished_len.load(Ordering::SeqCst)
    }

    pub fn get_writer(&self) -> Writer<T> {
        self.try_get_writer().expect("Cannot get Writer because one already exists")
    }

    pub fn try_get_writer(&self) -> Option<Writer<T>> {
        match self.inner.has_writer.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => Some(Writer {
                inner: self.inner.clone(),
                write_offset: self.inner.write_offset.load(Ordering::SeqCst),
                amount_written: 0,
            }),
            Err(_) => None
        }
    }
}

impl<T> std::fmt::Debug for MarchingBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

pub struct Reader<T> {
    inner: Arc<InnerMarchingBuffer<T>>,
    read_offset: usize,
    read_len: usize
}

impl<T> Reader<T> {
    pub fn access(&self) -> ReaderAccess<T> {
        self.try_access().expect("Cannot access Reader because concurrent Writer is already accessed")
    }

    pub fn try_access(&self) -> Option<ReaderAccess<T>> {
        match self.inner.data.try_read() {
            Ok(data) => {
                Some(ReaderAccess {
                    reader: self,
                    data,
                    read_offset: self.read_offset,
                    read_len: self.read_len,
                })
            },
            Err(_) => {
                None
            }
        }
    }
}

impl<T> Drop for Reader<T> {
    fn drop(&mut self) {
        self.inner.readers.fetch_sub(1, Ordering::SeqCst);
        self.inner.check_reset();
    }
}

impl<T> Clone for Reader<T> {
    fn clone(&self) -> Self {
        self.inner.readers.fetch_add(1, Ordering::SeqCst);
        Self {
            inner: self.inner.clone(),
            read_offset: self.read_offset,
            read_len: self.read_len
        }
    }
}

impl<T> std::fmt::Debug for Reader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("read_offset", &self.read_offset)
            .field("read_len", &self.read_len)
            .finish()
    }
}

pub struct ReaderAccess<'reader, 'data, T> {
    reader: &'reader Reader<T>,
    data: RwLockReadGuard<'data, Vec<T>>,
    // Stored in addition to the identically named fields in Reader so that ReaderAccess can implement Read, which mutates the Read struct.
    read_offset: usize,
    read_len: usize
}

impl<'reader, 'data, T> ReaderAccess<'reader, 'data, T> {
    pub fn as_slice(&self) -> &[T] {
        &self.data[self.read_offset .. (self.read_offset + self.read_len)]
    }

    pub fn is_empty(&self) -> bool {
        self.read_len == 0
    }
}

impl<'reader, 'data, T> std::fmt::Debug for ReaderAccess<'reader, 'data, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReaderAccess")
            .field("reader_offset", &self.reader.read_offset)
            .field("reader_len", &self.reader.read_len)
            .field("access_offset", &self.read_offset)
            .field("access_len", &self.read_len)
            .finish()
    }
}

impl<'reader, 'data> std::io::Read for ReaderAccess<'reader, 'data, u8> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let amount_read = std::cmp::min(self.read_len, buf.len());
        buf.copy_from_slice(&self.data.as_slice()[self.reader.read_offset .. (self.reader.read_offset + amount_read)]);
        self.read_offset += amount_read;
        self.read_len -= amount_read;
        Ok(amount_read)
    }
}

impl<'reader, 'data, T> Deref for ReaderAccess<'reader, 'data, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

pub struct Writer<T> {
    inner: Arc<InnerMarchingBuffer<T>>,
    // Where in the data Vec the writable data starts at.
    write_offset: usize,
    // How much has been written into this Write.
    amount_written: usize,
}

impl<T> Writer<T> {
    pub fn finish(&mut self) -> Reader<T> {
        let reader = Reader {
            inner: self.inner.clone(),
            read_offset: self.write_offset,
            read_len: self.amount_written,
        };
        self.inner.readers.fetch_add(1, Ordering::SeqCst);
        self.inner.write_offset.fetch_add(self.amount_written, Ordering::SeqCst);
        self.inner.finished_len.fetch_add(self.amount_written, Ordering::SeqCst);
        self.write_offset += self.amount_written;
        self.amount_written = 0;
        reader
    }

    pub fn access(&mut self) -> WriterAccess<T> {
        self.try_access().expect("Cannot access Writer because at least one concurrent Reader is already accessed")
    }

    pub fn try_access(&mut self) -> Option<WriterAccess<T>> {
        Some(WriterAccess {
            data: self.inner.data.try_write().ok()?,
            write_offset: &mut self.write_offset,
            amount_written: &mut self.amount_written,
        })
    }
}

impl<T: Default + Copy> Writer<T> {
    // A necessary convenience method for copying the contents of a Reader<T> into a Writer<T>. This method internally uses a temporary buffer
    // of size COPY_BUFFER_SIZE, which is needed because the underlying data buffer may be reallocated at any point during the copy.
    pub fn copy_from<const COPY_BUFFER_SIZE: usize>(&mut self, reader: &Reader<T>) {
        // Technically we would be able to void the double copying if there's sufficient capacity in the data buffer not to need a reallocation.
        // Or, if we just ensure() that there's enough additional capacity ahead of time. We could then use std::ptr::copy_nonoverlapping, but
        // that would require unsafe.
        // Or, I wonder if we could use various slice split() methods to get the disjoint slices without requiring unsafe?
        let mut copy_buffer = [T::default(); COPY_BUFFER_SIZE];
        let mut bytes_copied = 0;
        let mut bytes_remaining = reader.access().len();
        while bytes_remaining > 0 {
            let copied_this_round = std::cmp::min(bytes_remaining, 4096);
            &mut copy_buffer[..copied_this_round].copy_from_slice(&reader.access().as_slice()[bytes_copied .. (bytes_copied + copied_this_round)]);
            self.access().extend_from_slice(&copy_buffer[..copied_this_round]);
            bytes_copied += copied_this_round;
            bytes_remaining -= copied_this_round;
        }
    }
}

impl<T> Drop for Writer<T> {
    fn drop(&mut self) {
        self.inner.has_writer.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .expect("has_writer was false somehow when Writer was dropped");
        self.inner.check_reset();
    }
}

impl<T> std::fmt::Debug for Writer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer")
            .field("write_offset", &self.write_offset)
            .field("amount_written", &self.amount_written)
            .finish()
    }
}

pub struct WriterAccess<'writer, 'data, T> {
    data: RwLockWriteGuard<'data, Vec<T>>,
    write_offset: &'writer mut usize,
    amount_written: &'writer mut usize,
}

impl<'writer, 'data, T> WriterAccess<'writer, 'data, T> {
    pub fn as_slice(&self) -> &[T] {
        &self.data[*self.write_offset .. (*self.write_offset + *self.amount_written)]
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[*self.write_offset .. (*self.write_offset + *self.amount_written)]
    }

    pub fn push(&mut self, value: T) {
        self.data.push(value);
        *self.amount_written += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if *self.amount_written > 0 {
            *self.amount_written -= 1;
            self.data.pop()
        } else {
            None
        }
    }
}

impl<'writer, 'data, T: Clone> WriterAccess<'writer, 'data, T> {
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.data.extend_from_slice(slice);
        *self.amount_written += slice.len();
    }
}

impl<'writer, 'data> std::fmt::Write for WriterAccess<'writer, 'data, u8> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.data.extend_from_slice(s.as_bytes());
        *self.amount_written += s.len();
        Ok(())
    }
}

impl<'writer, 'data> std::io::Write for WriterAccess<'writer, 'data, u8> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.data.extend_from_slice(buf);
        *self.amount_written += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'writer, 'data, T> Deref for WriterAccess<'writer, 'data, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'writer, 'data, T> DerefMut for WriterAccess<'writer, 'data, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    #[test]
    fn basic_nesting_test() {
        let alloc = MarchingBuffer::new();
        {
            let mut writer = alloc.get_writer();

            write!(writer.access(), "Hello world").unwrap();
            let hello_world = writer.finish();
            assert_eq!(b"Hello world", hello_world.access().as_slice());
            assert_eq!("Hello world".len(), alloc.finished_len());

            write!(writer.access(), "Foo").unwrap();
            // "Foo" not counted to length until the current write is finished.
            assert_eq!("Hello world".len(), alloc.finished_len());

            write!(writer.access(), "Bar").unwrap();
            let foo_bar = writer.finish();
            assert_eq!(b"FooBar", foo_bar.access().as_slice());
            assert_eq!("Hello world".len() + "FooBar".len(), alloc.finished_len());

            write!(writer.access(), "End of line").unwrap();
            writer.finish();
            assert_eq!("Hello world".len() + "FooBar".len() + "End of line".len(), alloc.finished_len());
        }
        assert_eq!(0, alloc.finished_len());
    }

    #[test]
    fn unfinished_writes_are_ignored() {
        let alloc = MarchingBuffer::new();
        {
            let mut writer = alloc.get_writer();
            write!(writer.access(), "Hello world").unwrap();
        }
        {
            let mut writer = alloc.get_writer();
            write!(writer.access(), "foo bar").unwrap();
            assert_eq!(b"foo bar", writer.finish().access().as_slice());
        }
        assert_eq!(0, alloc.finished_len());
    }
}
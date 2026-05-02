//! Remote session protocol (Tier 3 / claude-code parity).
//!
//! Mirrors `services/remote/`. JSON-RPC-2.0-over-stdio that lets an
//! external host (e.g., a TUI in Zode, an editor extension) drive an
//! agent runtime running in a separate process. Useful for:
//!
//! - Decoupling UI process lifecycle from agent process lifecycle.
//! - Sandboxing the agent into a different security domain.
//! - Sharing one agent between multiple UI surfaces.
//!
//! This module ships:
//!
//! - [`protocol`] — wire types ([`RpcRequest`], [`RpcResponse`],
//!   [`RpcNotification`], [`RpcError`]).
//! - [`session`] — the [`RemoteSession`] trait the host implements
//!   to handle the agent end of the protocol.
//! - [`codec`] — line-delimited JSON codec helpers (read one frame
//!   from a buffer, encode one frame to bytes).
//!
//! The actual stdio runner lives in the host (Zode); this module
//! provides the wire format + a clean trait surface so the runner is
//! a thin loop.

pub mod codec;
pub mod protocol;
pub mod session;

pub use codec::{decode_frame, encode_frame, FrameError};
pub use protocol::{
    method, RpcError, RpcErrorCode, RpcId, RpcMessage, RpcNotification, RpcRequest, RpcResponse,
};
pub use session::{RemoteSession, SessionError};

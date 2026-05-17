/**
 * Server-side Supabase client + database type definitions.
 *
 * The service-role client is created lazily and cached per process. It is
 * imported only from server contexts (API routes, server components) — never
 * from "use client" files. Next.js will throw a build-time error if the
 * service-role key is imported into client code because the env var lacks the
 * `NEXT_PUBLIC_` prefix.
 *
 * Types here mirror the canonical schema in
 * `supabase/migrations/20260514120000_initial_schema.sql`. When the schema
 * changes, update these by hand or regenerate via the Supabase CLI:
 *   npx supabase gen types typescript --linked > lib/db-types.ts
 */

import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// ── DB types ──────────────────────────────────────────────────────────────

export type AgentLayer =
  | "coordination"
  | "creative"
  | "generation"
  | "editing"
  | "quality"
  | "delivery";

export type AgentState = "idle" | "processing" | "error";

export type ServiceType =
  | "thumbnail"
  | "social_graphic"
  | "headshot"
  | "background_removal"
  | "logo"
  | "business_design";

export type OrderStatus =
  | "pending"
  | "clarification_needed"
  | "awaiting_response"
  | "processing"
  | "qc"
  | "ready_for_delivery"
  | "delivered"
  | "error"
  | "cancelled";

export type PackageStatus =
  | "pending_approval"
  | "approved"
  | "sent_to_fiverr"
  | "rejected";

export interface Agent {
  id: string;
  agent_key: string;
  display_name: string;
  layer: AgentLayer;
  description: string;
  layer_order: number;
  position_x: number;
  position_y: number;
  handles_service_types: ServiceType[];
  is_active: boolean;
}

export interface AgentStatus {
  agent_id: string;
  current_status: AgentState;
  current_order_id: string | null;
  current_run_id: string | null;
  last_log: string | null;
  last_completed_at: string | null;
  last_error_at: string | null;
  total_runs: number;
  total_errors: number;
  updated_at: string;
}

export interface Order {
  id: string;
  fiverr_order_id: string | null;
  service_type: ServiceType;
  client_username: string | null;
  brief: string;
  status: OrderStatus;
  confidence_score: number | null;
  created_at: string;
  updated_at: string;
}

export interface DeliveryPackage {
  id: string;
  order_id: string;
  zip_url: string | null;
  delivery_message: string;
  upsell_suggestion: string | null;
  status: PackageStatus;
  rejection_reason: string | null;
  created_at: string;
}

// ── Client construction ───────────────────────────────────────────────────

let _serverClient: SupabaseClient | null = null;

/**
 * Get the server-side Supabase client (service-role).
 * Server-only — DO NOT import from a "use client" file.
 */
export function getServerClient(): SupabaseClient {
  if (_serverClient) return _serverClient;

  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error(
      "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the dashboard env.",
    );
  }

  _serverClient = createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
  return _serverClient;
}
